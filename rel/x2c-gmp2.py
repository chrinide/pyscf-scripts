#!/usr/bin/env python

import numpy
from pyscf import gto, scf, x2c
from pyscf import ao2mo, lib, mp

mol = gto.Mole()
mol.basis = 'dzp-dk'
mol.atom = '''
O
H 1 1.1
H 1 1.1 2 104
'''
mol.charge = 0
mol.spin = 0
mol.symmetry = 1
mol.verbose = 4
mol.build()

mf = scf.RHF(mol)
enr = mf.kernel()

pt2 = mp.MP2(mf)
pt2.kernel()

mf = x2c.UHF(mol)
dm = mf.get_init_guess() + 0.0j
ex2c = mf.kernel(dm)

lib.logger.info(mf,'**** Relativistic GMP2')
eri_ao = mol.intor('int2e_spinor')
nao = eri_ao.shape[0]
nocc = mol.nelectron
nvir = nao - nocc
co = mf.mo_coeff[:,:nocc]
cv = mf.mo_coeff[:,nocc:]
eri_mo = ao2mo.general(eri_ao,(co,cv,co,cv)).reshape(nocc,nvir,nocc,nvir)
eri_mo = eri_mo - eri_mo.transpose(0,3,2,1)
eo = mf.mo_energy[:nocc]
ev = mf.mo_energy[nocc:]
e_denom = 1.0/(-ev.reshape(-1,1,1,1)+eo.reshape(-1,1,1)-ev.reshape(-1,1)+eo)
t2 = numpy.zeros((nvir,nocc,nvir,nocc))
t2 = eri_mo.transpose(1,0,3,2)*e_denom 
e_mp2 = 0.25*numpy.einsum('iajb,aibj->', eri_mo, t2, optimize=True)
lib.logger.info(mf,"!*** E(MP2): %s" % e_mp2)
lib.logger.info(mf,"!*** E(X2C+MP2): %s" % (e_mp2+ex2c))

