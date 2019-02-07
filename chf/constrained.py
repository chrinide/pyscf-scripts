#!/usr/bin/env python

#### Warning en casos correlacionados hay que cambiar
#### fock por la funcion hcore y meter la ligadura para
#### probar mas que de sobra

import numpy, h5py
from pyscf import gto, scf, dft, lib
from pyscf.tools import wfn_format

subname = 'h2'
name = 'h2_ct_0p2'
atms = [0,1]
natm = len(atms)
mult = [0.00,0.2]

mol = gto.Mole()
mol.atom = '''
H      0.000000      0.000000      0.000000
H      0.000000      0.000000      0.750000
'''
mol.basis = '3-21g'
mol.verbose = 4
mol.spin = 0
mol.symmetry = 0
mol.charge = 0
mol.incore_anyway = True
mol.build()

# Read overlap matrix and transform to AO basis
nao = mol.nao_nr()
saom = numpy.zeros((natm,nao,nao))
mol = lib.chkfile.load_mol(subname+'.chk')
mo_coeff = scf.chkfile.load(subname+'.chk', 'scf/mo_coeff')
mo_occ = scf.chkfile.load(subname+'.chk', 'scf/mo_occ')
coeff = numpy.linalg.inv(mo_coeff)
with h5py.File(subname+'.chk.h5') as f:
    for i in range(natm):
        ia = atms[i]
        idx = 'ovlp'+str(ia)
        saom[i] = f[idx+'/aom'].value
	saom[i] = coeff.T.dot(saom[i]).dot(coeff)

def get_pop(s, mult):
    nao = s.shape[1]
    fock = numpy.zeros((nao,nao))
    for ia in range(natm):
        fock += mult[ia]*s[ia]
    return fock

def get_cfock(h1e, s1e, vhf, dm, cycle=0, mf_diis=None):
    fock = old_get_fock(h1e, s1e, vhf, dm, cycle, mf_diis)
    fock -= get_pop(saom, mult) 
    return fock

diis_obj = scf.ADIIS()
diis_obj.diis_space = 24
diis_obj.diis_start_cycle = 14

# Guess
mf = scf.RHF(mol)
mf.verbose = 0
mf.kernel()
dm = mf.make_rdm1()

# Calc
mf = scf.RHF(mol)#.newton()
mf.conv_tol = 1e-6
mf.max_cycle = 120
mf.diis = diis_obj
old_get_fock = mf.get_fock
mf.get_fock = get_cfock 
mf.kernel(dm)
mf.analyze(with_meta_lowdin=False)
dm = mf.make_rdm1()
for i in range(natm):
    ia = atms[i]
    pop = numpy.einsum('ij,ji->',saom[i],dm)
    lib.logger.info(mol,'Population atom %d : %f' % (ia,pop))

wfn_file = name + '.wfn'
idx = mf.mo_occ>0
with open(wfn_file, 'w') as f2:
    wfn_format.write_mo(f2, mol, mf.mo_coeff[:,idx], \
    mo_occ=mf.mo_occ[idx], mo_energy=mf.mo_energy[idx])

