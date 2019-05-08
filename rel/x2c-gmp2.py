#!/usr/bin/env python

import numpy
from pyscf import gto, scf, x2c
from pyscf import ao2mo, lib, mp
einsum = lib.einsum

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

mf = scf.GHF(mol).x2c()
mf.verbose = 0
enr = mf.kernel()

lib.logger.info(mol,'**** Relativistic Scalar MP2')
pt2 = mp.MP2(mf)
pt2.verbose = 4
pt2.kernel()

mf = x2c.UHF(mol)
dm = mf.get_init_guess() + 0.1j
mf.verbose = 0
ex2c = mf.kernel(dm)

lib.logger.info(mol,'**** Relativistic GMP2')
oidx = mf.mo_occ>0
vidx = mf.mo_occ<=0
eri_ao = mol.intor('int2e_spinor')
nao = eri_ao.shape[0]
nocc = mol.nelectron
nvir = nao - nocc
nmo = nocc + nvir
co = mf.mo_coeff[:,oidx]
cv = mf.mo_coeff[:,vidx]
c = numpy.hstack((co,cv))
o = slice(0, nocc)
v = slice(nocc, None)
#eri_mo = ao2mo.general(eri_ao,(c,c,c,c)).reshape(nmo,nmo,nmo,nmo)
eri_mo = einsum('pi,pqrs->iqrs', c.conj(), eri_ao)
eri_mo = einsum('qa,iqrs->iars', c, eri_mo)
eri_mo = einsum('iars,rj->iajs', eri_mo, c.conj())
eri_mo = einsum('iajs,sb->iajb', eri_mo, c)
eri_mo = eri_mo - eri_mo.transpose(0,3,2,1)
eo = mf.mo_energy[:nocc]
ev = mf.mo_energy[nocc:]
e_denom = 1.0/(-ev.reshape(-1,1,1,1)+eo.reshape(-1,1,1)-ev.reshape(-1,1)+eo)
t2 = numpy.zeros((nvir,nocc,nvir,nocc), dtype=numpy.complex128)
t2 = eri_mo[v,o,v,o]*e_denom 
e_mp2 = 0.25*numpy.einsum('iajb,aibj->', eri_mo[o,v,o,v], t2, optimize=True)
lib.logger.info(mol,"!*** E(MP2): %s" % e_mp2)
lib.logger.info(mol,"!*** E(X2C+MP2): %s" % (e_mp2+ex2c))

