#!/usr/bin/env python

import numpy
from pyscf import gto, scf
from pyscf import ao2mo, lib
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

mf = scf.DRHF(mol)
dm = mf.get_init_guess() + 0.1j
mf.with_ssss = True
e4c = mf.kernel(dm)

n4c, nmo = mf.mo_coeff.shape
n2c = n4c // 2
nN = nmo // 2
nocc = int(mf.mo_occ.sum())
nP = nN + nocc
nvir = nmo - nP
c = lib.param.LIGHT_SPEED

print('4C-MP2 negative energy states are removed')
coeffl = mf.mo_coeff[:n2c,nN:] 
coeffs = mf.mo_coeff[n2c:,nN:]*(0.5/c) 
eri_llll = mol.intor('int2e_spinor')
eri_llll = einsum('pi,pqrs->iqrs', coeffl.conj(), eri_llll)
eri_llll = einsum('qa,iqrs->iars', coeffl, eri_llll)
eri_llll = einsum('iars,rj->iajs', eri_llll, coeffl.conj())
eri_llll = einsum('iajs,sb->iajb', eri_llll, coeffl)

eri_ssss = mol.intor('int2e_spsp1spsp2_spinor')
eri_ssss = einsum('pi,pqrs->iqrs', coeffs.conj(), eri_ssss)
eri_ssss = einsum('qa,iqrs->iars', coeffs, eri_ssss)
eri_ssss = einsum('iars,rj->iajs', eri_ssss, coeffs.conj())
eri_ssss = einsum('iajs,sb->iajb', eri_ssss, coeffs)

eri_ssll = mol.intor('int2e_spsp1_spinor')
eri_ssll = einsum('pi,pqrs->iqrs', coeffs.conj(), eri_ssll)
eri_ssll = einsum('qa,iqrs->iars', coeffs, eri_ssll)
eri_ssll = einsum('iars,rj->iajs', eri_ssll, coeffl.conj())
eri_ssll = einsum('iajs,sb->iajb', eri_ssll, coeffl)

eri_llss = mol.intor('int2e_spsp2_spinor')
eri_llss = einsum('pi,pqrs->iqrs', coeffl.conj(), eri_llss)
eri_llss = einsum('qa,iqrs->iars', coeffl, eri_llss)
eri_llss = einsum('iars,rj->iajs', eri_llss, coeffs.conj())
eri_llss = einsum('iajs,sb->iajb', eri_llss, coeffs)
o = slice(0, nocc)
v = slice(nocc, None)
eri_mo = eri_llll + eri_ssss + eri_llss + eri_ssll
eri_mo = eri_mo - eri_mo.transpose(0,3,2,1)
eo = mf.mo_energy[nN:nP]
ev = mf.mo_energy[nP:]
e_denom = 1.0/(-ev.reshape(-1,1,1,1)+eo.reshape(-1,1,1)-ev.reshape(-1,1)+eo)
t2 = numpy.zeros((nvir,nocc,nvir,nocc), dtype=numpy.complex128)
t2 = eri_mo[v,o,v,o]*e_denom 
e_mp2 = 0.25*numpy.einsum('iajb,aibj->', eri_mo[o,v,o,v], t2, optimize=True)
lib.logger.info(mf,"!*** E(MP2): %s" % e_mp2)
lib.logger.info(mf,"!*** E(X2C+MP2): %s" % (e_mp2+e4c))

