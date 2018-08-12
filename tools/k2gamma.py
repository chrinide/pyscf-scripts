#!/usr/bin/env python

'''
Convert the k-sampled MO to corresponding Gamma-point supercell MO.
Zhihao Cui zcui@caltech.edu
'''

import numpy as np
import scipy
from scipy import linalg as la
import cmath, os, sys, copy

from pyscf import scf, gto, lo, lib, tools
from pyscf.lib import numpy_helper as np_helper
from pyscf.pbc import gto as pgto
from pyscf.pbc import scf as pscf
from pyscf.pbc import df
from pyscf.pbc.tools import pbc as tools_pbc


def get_R_vec(cell, abs_kpts, kmesh):

    '''
    get supercell R vector based on k mesh
    '''

    latt_vec = cell.lattice_vectors()
    R_rel_a = np.arange(kmesh[0]) 
    R_rel_b = np.arange(kmesh[1]) 
    R_rel_c = np.arange(kmesh[2])
    R_rel_mesh = np_helper.cartesian_prod((R_rel_a, R_rel_b, R_rel_c))
    R_abs_mesh = np.einsum('nu, uv -> nv', R_rel_mesh, latt_vec)
    return R_abs_mesh

def find_degenerate(mo_energy, mo_coeff, real_split = False, tol = 1e-5):

    '''
    split the mo_energy into groups based on degenracy.
    further split the real MO out by set real_split to be True
    mo_energy should range from lowest to highest.
    return grouped mo_energy and its indices.    
    '''

    real_tol = tol * 0.01 # tol for split real
    
    res = []
    res_idx = []
    g_cur = [mo_energy[0]]
    idx_cur = [0]

    for i in xrange(1, len(mo_energy)):
        diff = mo_energy[i] - mo_energy[i-1]
        if diff < tol:
            g_cur.append(mo_energy[i])
            idx_cur.append(i)
        else:
            res.append(g_cur)
            res_idx.append(idx_cur)
            g_cur = [mo_energy[i]]
            idx_cur = [i]
        if i == len(mo_energy)-1:
            res.append(g_cur)
            res_idx.append(idx_cur)

    if real_split:
        res_idx_new = []
        res_new = []
        for i in xrange(len(res_idx)):
            res_idx_tmp = copy.deepcopy(res_idx[i])
            res_tmp = copy.deepcopy(res[i])
            
            tmp = 0            

            for j in xrange(len(res_idx_tmp)):
                if la.norm(mo_coeff[:,res_idx_tmp[j-tmp]].imag) < real_tol:
                    p = res_idx_tmp.pop(j-tmp)
                    res_idx_new.append([p])

                    e = res_tmp.pop(j-tmp)
                    res_new.append([e])

                    tmp += 1

            if res_idx_tmp != []:
                    res_idx_new.append(res_idx_tmp) 
                    res_new.append(res_tmp)
                
        # sort again to make sure the slightly lower energy state to be the first
        sort_idx = sorted(range(len(res_new)), key=lambda k: res_new[k])
        res_new = [res_new[i] for i in sort_idx]
        res_idx_new = [res_idx_new[i] for i in sort_idx]
        res_idx = res_idx_new
        res = res_new

    return res, res_idx


def k2gamma(kmf, abs_kpts, kmesh, realize = True, real_split = False, tol_deg = 5e-5):

    '''
    convert the k-sampled mo coefficient to corresponding supercell gamma-point mo coefficient.
    set realize = True to make sure the final wavefunction to be real.
    return the supercell gamma point object 
    math:
         C_{\nu ' n'} = C_{\vecR\mu, \veck m} = \qty[ \frac{1}{\sqrt{N_{\UC}}} \e^{\ii \veck\cdot\vecR} C^{\veck}_{\mu  m}]
    '''

    #np.set_printoptions(4,linewidth=1000)

    R_abs_mesh = get_R_vec(kmf.cell, abs_kpts, kmesh)    
    phase = np.exp(1j*np.einsum('Ru, ku -> Rk',R_abs_mesh, abs_kpts))
    
    E_k = np.asarray(kmf.mo_energy)
    occ_k = np.asarray(kmf.mo_occ)
    C_k = np.asarray(kmf.mo_coeff)
    
    Nk, Nao, Nmo = C_k.shape
    NR = R_abs_mesh.shape[0]    

    C_gamma = np.einsum('Rk, kum -> Rukm', phase, C_k) / np.sqrt(NR)
    C_gamma = C_gamma.reshape((NR, Nao, Nk*Nmo))

    # sort energy of km
    E_k_flat = E_k.flatten()
    E_k_sort_idx = np.argsort(E_k_flat)
    E_k_sort = E_k_flat[E_k_sort_idx]
    occ_sort = occ_k.flatten()[E_k_sort_idx]
    
    C_gamma = C_gamma[:, :, E_k_sort_idx]
    C_gamma = C_gamma.reshape((NR*Nao, Nk*Nmo))

    # supercell object
    sc = tools_pbc.super_cell(kmf.cell, kmesh)
    sc.verbose = 0
    kmf_sc = pscf.KRHF(sc, [[0.0,0.0,0.0]])
    kmf_sc.with_df = df.FFTDF(sc, [[0.0, 0.0, 0.0]]) 
    S_sc = kmf_sc.get_ovlp()[0].real
   
    # make MO to be real
    if realize:
        
        real_tol = tol_deg # tolerance of residue of real or imag part
        null_tol = min(tol_deg * 10.0, 1.0e-3) # tolerance of 0 for nat_orb selection 

        print "Realize the gamma point MO ..." 
        C_gamma_real = np.zeros_like(C_gamma, dtype = np.double)

        res, res_idx = find_degenerate(E_k_sort, C_gamma, real_split = real_split, tol = tol_deg )
        print "Energy spectrum group:", res
        print "Energy idx:", res_idx
        col_idx = 0
        for i, gi_idx in enumerate(res_idx):
            gi = C_gamma[:,gi_idx]

            # using dm to solve natural orbitals, to make the orbitals real
            dm =  gi.dot(gi.conj().T)
            if la.norm(dm.imag) > real_tol:
                print "density matrix of converted Gamma MO has large imaginary part."
                sys.exit(1)
            eigval, eigvec = la.eigh(dm.real, S_sc, type = 2)
            nat_orb = eigvec[:, eigval > null_tol]
            if nat_orb.shape[1] != len(gi_idx):
                print "Realization error, not find correct number of linear combination coefficient"
                sys.exit(1)
            for j in xrange(nat_orb.shape[1]):
                C_gamma_real[:,col_idx] = nat_orb[:, j]
                col_idx += 1

        C_gamma = C_gamma_real
    
    # save to kmf_sc obj
    kmf_sc.mo_coeff = [C_gamma]
    kmf_sc.mo_energy = [np.asarray(E_k_sort)]
    kmf_sc.mo_occ = [np.asarray(occ_sort)]

    return kmf_sc

            
if __name__ == '__main__':
    
    np.set_printoptions(3,linewidth=1000)

    cell = pgto.Cell()
    cell.spin = 0
    cell.symmetry = 0
    cell.charge = 0
    cell.verbose = 4
    cell.a = '''
      0.000000000000000   2.014000000000000   2.014000000000000
      2.014000000000000   0.000000000000000   2.014000000000000
      2.014000000000000   2.014000000000000   0.000000000000000
    '''
    cell.atom = '''
    Li  0.000000000000000   0.000000000000000   0.000000000000000 
    F   2.014000000000000   2.014000000000000   2.014000000000000 
    '''
    cell.basis = 'gth-szv'
    cell.pseudo = 'gth-pade'
    cell.build()
    
    kmesh = [2, 2, 2]
    abs_kpts = cell.make_kpts(kmesh)
    abs_kpts -= abs_kpts[0]
    kmf = pscf.KRHF(cell, abs_kpts)
    kmf.with_df = df.FFTDF(cell, abs_kpts)
    kmf.verbose = 4
    ekpt = kmf.run()

    kmf_sc = k2gamma(kmf, abs_kpts, kmesh, realize=True, tol_deg=5e-5, real_split=False)
    c_g_ao = kmf_sc.mo_coeff[0] 
    dm_convert = 2.0*c_g_ao[:, kmf_sc.mo_occ[0]>0].dot(c_g_ao[:, kmf_sc.mo_occ[0]>0].conj().T)

    mo_coeff_occ = kmf_sc.mo_coeff[0][:,kmf_sc.mo_occ[0]>0]
    lo_iao = lo.iao.iao(kmf_sc.cell, mo_coeff_occ)
    lo_iao = lo.vec_lowdin(lo_iao, kmf_sc.get_ovlp()[0])
    mo_coeff_occ = reduce(np.dot, (lo_iao.conj().T, kmf_sc.get_ovlp()[0], mo_coeff_occ))
    dm = np.dot(mo_coeff_occ, mo_coeff_occ.conj().T)*2.0
    pmol = kmf_sc.cell.copy()
    pmol.build(False, False, basis='minao')
    kmf_sc.mulliken_pop(pmol, dm, s=np.eye(pmol.nao_nr()))

    # The following is to check whether the MO is correctly coverted: 
    sc = tools_pbc.super_cell(cell, kmesh)
    kmf_sc2 = pscf.KRHF(sc, [[0.0, 0.0, 0.0]])
    kmf_sc2.with_df = df.FFTDF(sc, [[0.0, 0.0, 0.0]]) 
    s = kmf_sc2.get_ovlp()[0]

    print "Run supercell gamma point calculation..." 
    ekpt_sc = kmf_sc2.run([dm_convert])
    sc_mo = kmf_sc2.mo_coeff[0]

    # lowdin of sc_mo and c_g_ao
    sc_mo_o = la.sqrtm(s).dot(sc_mo)
    c_g = la.sqrtm(s).dot(c_g_ao)
    res, res_idx = find_degenerate(kmf_sc.mo_energy[0], c_g, real_split=False, tol=5e-5)

    print res
    print kmf_sc2.mo_energy
    print c_g_ao.conj().T.dot(s).dot(c_g_ao)

    #for i in xrange(len(res_idx)): 
    #    print 
    #    print "subspace:", i
    #    print "index:", res_idx[i]
    #    print "energy:", res[i]
    #    u, sigma, v = la.svd(c_g[:,res_idx[i]].T.conj().dot(sc_mo_o[:,res_idx[i]]))
    #    print "singular value of subspace (C_convert * C_calculated):" , sigma
        
