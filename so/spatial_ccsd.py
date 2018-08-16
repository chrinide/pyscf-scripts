#!/usr/bin/env python

import numpy
from pyscf import gto, scf, cc, lib, ao2mo
from pyscf.tools import molden
einsum = lib.einsum

name = 'uccsd'

mol = gto.Mole()
mol.verbose = 4
mol.atom = '''
O      0.000000      0.000000      0.118351
H      0.000000      0.761187     -0.469725
H      0.000000     -0.761187     -0.469725
'''
mol.basis = 'sto-3g'
mol.symmetry = 1
mol.spin = 0
mol.charge = 0
mol.build()

mf = scf.UHF(mol)
mf.scf()

frozen = [[0],[0]] # 1sa and 1sb
mcc = cc.UCCSD(mf)
mcc.frozen = frozen
mcc.diis_space = 10
mcc.conv_tol = 1e-6
mcc.conv_tol_normt = 1e-6
mcc.max_cycle = 150
mcc.kernel()

rdm1a, rdm1b = mcc.make_rdm1()
rdm2aa, rdm2ab, rdm2bb = mcc.make_rdm2()

rdm1 = rdm1a + rdm1b
rdm2 = rdm2aa + rdm2ab + rdm2ab.transpose(2,3,0,1) + rdm2bb

occ_a = mf.mo_occ[0]
occ_b = mf.mo_occ[1]
mo_a = mf.mo_coeff[0]
mo_b = mf.mo_coeff[1]
nmoa = mo_a.shape[1]
nmob = mo_b.shape[1]

eriaa = ao2mo.kernel(mf._eri, mo_a, compact=False).reshape([nmoa]*4)
eribb = ao2mo.kernel(mf._eri, mo_b, compact=False).reshape([nmob]*4)
eriab = ao2mo.kernel(mf._eri, (mo_a,mo_a,mo_b,mo_b), compact=False)
eriab = eriab.reshape([nmoa,nmoa,nmob,nmob])
hcore = mf.get_hcore()
h1a = reduce(numpy.dot, (mo_a.T.conj(), hcore, mo_a))
h1b = reduce(numpy.dot, (mo_b.T.conj(), hcore, mo_b))
e1 = einsum('ij,ji', h1a, rdm1a)
e1+= einsum('ij,ji', h1b, rdm1b)
e1+= einsum('ijkl,ijkl', eriaa, rdm2aa)*0.5
e1+= einsum('ijkl,ijkl', eriab, rdm2ab)
e1+= einsum('ijkl,ijkl', eribb, rdm2bb)*0.5
e1+= mol.energy_nuc()
lib.logger.info(mcc,"* Energy with spin 1/2-RDM : %.8f" % e1) 

if (mol.spin==0):
    e1 = einsum('ij,ji', h1a, rdm1)
    e1+= einsum('ijkl,ijkl', eriaa, rdm2)*0.5
    e1+= mol.energy_nuc()
    lib.logger.info(mcc,"* Energy with spatial 1/2-RDM : %.8f" % e1) 

