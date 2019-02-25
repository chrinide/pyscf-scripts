#!/usr/bin/env python

import numpy, utils
from pyscf import lib, ao2mo
from pyscf.pbc import gto, scf, df, mp
from pyscf.tools import wfn_format

name = 'mp2'

cell = gto.Cell()
cell.unit = 'A'
cell.a = [[4,0,0],[0,1,0],[0,0,1]]
cell.atom = '''
H       0.0000000   0.0000000   0.0000000
H       1.0000000   0.0000000   0.0000000
H       2.0000000   0.0000000   0.0000000
H       3.0000000   0.0000000   0.0000000
''' 
cell.basis = 'sto-6g'
cell.dimension = 1
cell.verbose = 4
cell.build()
 
gdf = df.GDF(cell)

mf = scf.RHF(cell)
mf.with_df = gdf
mf.with_df.auxbasis = 'cc-pvdz-jkfit'
mf.max_cycle = 150
mf.chkfile = name+'.chk'
#mf.with_df._cderi_to_save = name+'_eri.h5'
#mf.with_df._cderi = name+'_eri.h5' 
#mf = mole_scf.addons.remove_linear_dep_(mf)
#mf.__dict__.update(scf.chkfile.load(name+'.chk', 'scf'))
#dm = mf.make_rdm1()
ehf = mf.kernel()

ncore = 0 
nao, nmo = mf.mo_coeff.shape
nocc = cell.nelectron//2 - ncore
nvir = nmo - nocc - ncore
mo_core = mf.mo_coeff[:,:ncore]
mo_occ = mf.mo_coeff[:,ncore:ncore+nocc]
mo_vir = mf.mo_coeff[:,ncore+nocc:]
co = mo_occ
cv = mo_vir
eo = mf.mo_energy[ncore:ncore+nocc]
ev = mf.mo_energy[ncore+nocc:]
lib.logger.info(mf,"\n+++ GAMMA point MP2 ")
lib.logger.info(mf,"* Core orbitals: %d" % ncore)
lib.logger.info(mf,"* Virtual orbitals: %d" % (len(ev)))

eri_mo = ao2mo.general(mf._eri, (co,cv,co,cv), compact=False)
eri_mo = eri_mo.reshape(nocc,nvir,nocc,nvir)
e_denom = 1.0/(eo.reshape(-1,1,1,1)-ev.reshape(-1,1,1)+eo.reshape(-1,1)-ev)
t2 = numpy.zeros((nocc,nvir,nocc,nvir))
t2 = 2.0*lib.einsum('iajb,iajb->iajb', eri_mo, e_denom)
t2 -= lib.einsum('ibja,iajb->iajb', eri_mo, e_denom)
e_mp2 = numpy.einsum('iajb,iajb->', eri_mo, t2, optimize=True)
lib.logger.info(mf,"!*** E(MP2): %12.8f" % e_mp2)
lib.logger.info(mf,"!**** E(HF+MP2): %12.8f" % (e_mp2+ehf))

wfn_file = name + '.wfn'
ao_loc = cell.ao_loc_nr()
fspt = open(wfn_file,'w')
wfn_format.write_mo(fspt, cell, mf.mo_coeff, mo_occ=mf.mo_occ, mo_energy=mf.mo_energy)
fspt.write('MP2\n')
fspt.write('1-RDM:\n')
occup = 2.0
norb = cell.nelectron//2
for i in range(norb):
    fspt.write('%i %i %.8f\n' % ((i+1), (i+1), occup))
fspt.write('t2_iajb:\n')
for i in range(nocc):
    for j in range(nvir):
        for k in range(nocc):
            for l in range(nvir):
                if (abs(t2[i,j,k,l]) > 1e-8):
                        fspt.write('%i %i %i %i %.10f\n' % ((i+1+ncore), \
                        (j+1+nocc+ncore), (k+1+ncore), (l+1+nocc+ncore), \
                        t2[i,j,k,l]*2.0))
a = cell.a
t = cell.get_lattice_Ls()
t = t[numpy.argsort(lib.norm(t, axis=1))]
kpts = numpy.asarray([0.0, 0.0, 0.0])
fspt.write('CRYSTAL\n')
fspt.write('GAMMA %11.8f %11.8f %11.8f\n' % (kpts[0], kpts[1], kpts[2]))
fspt.write('CELL\n')
fspt.write(' %11.8f %11.8f %11.8f\n' % (a[0][0], a[0][1], a[0][2]))
fspt.write(' %11.8f %11.8f %11.8f\n' % (a[1][0], a[1][1], a[1][2]))
fspt.write(' %11.8f %11.8f %11.8f\n' % (a[2][0], a[2][1], a[2][2]))
fspt.write('T-VECTORS %3d\n' % len(t))
for i in range(len(t)):
    fspt.write(' %11.8f %11.8f %11.8f\n' % (t[i][0], t[i][1], t[i][2]))
fspt.close()

#cv, ev = utils.getdffno(mf,ncore,eri_mo,thresh_vir=1e-4)
#lib.logger.info(mf,"* FNO GAMMA point MP2 ")
#nvir = len(ev)
#t2 = numpy.zeros((nocc,nvir,nocc,nvir))
#eri_mo = lib.einsum('rj,Qrs->Qjs', co, dferi)
#eri_mo = lib.einsum('sb,Qjs->Qjb', cv, eri_mo)
#vv_denom = -ev.reshape(-1,1)-ev
#for i in range(nocc):
#    eps_i = eo[i]
#    i_Qv = eri_mo[:, i, :].copy()
#    for j in range(nocc):
#        eps_j = eo[j]
#        j_Qv = eri_mo[:, j, :].copy()
#        viajb = lib.einsum('Qa,Qb->ab', i_Qv, j_Qv)
#        vibja = lib.einsum('Qb,Qa->ab', i_Qv, j_Qv)
#        div = 1.0 / (eps_i + eps_j + vv_denom)
#        t2[i,:,j,:] += 2.0*lib.einsum('ab,ab->ab', viajb, div) 
#        t2[i,:,j,:] -= 1.0*lib.einsum('ab,ab->ab', vibja, div) 
#e_mp2 = lib.einsum('iajb,Qia->Qjb', t2, eri_mo)
#e_mp2 = numpy.einsum('Qjb,Qjb->', e_mp2, eri_mo, optimize=True)
#lib.logger.info(mf,"!*** FNO E(MP2): %12.8f" % e_mp2)
#lib.logger.info(mf,"!**** FNO E(HF+MP2): %12.8f" % (e_mp2+ehf))

#pt = mp.RMP2(mf)
#pt.frozen = ncore
#pt.kernel()
