#!/usr/bin/env python

import numpy
from pyscf import gto, scf, lib
einsum = lib.einsum

mol = gto.Mole()
mol.basis = 'dzp-dk'
mol.atom = '''
Te
H 1 2.5
H 1 2.5 2 104
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
enuc = mol.energy_nuc() 
print('Nuclear energy : %s' % enuc)

dm = mf.make_rdm1()
rdm1 = numpy.eye(nocc, dtype=dm.dtype)

print('**** Info on AO basis')
s = mf.get_ovlp()
h = mf.get_hcore()
hcore = numpy.einsum('ij,ij->', dm, h)
pop = numpy.einsum('ij,ij->', dm, s)
print('Population : %s' % pop)
print('Hcore energy : %s' % hcore)
vj, vk = mf.get_jk()
bie1 = numpy.einsum('ij,ij->',vj,dm)*0.5 # J
bie2 = numpy.einsum('ij,ij->',vk,dm)*0.5 # XC
print('J energy : %s' % bie1)
print('XC energy : %s' % -bie2)
print('EE energy : %s' % (bie1-bie2))
etot = enuc + hcore + bie1 - bie2
print('Total energy : %s' % etot)

print('**** Info on MO spinor by L,S components')
s = mol.intor_symmetric('int1e_ovlp_spinor')
t = mol.intor_symmetric('int1e_spsp_spinor')
coeffl = mf.mo_coeff[:n2c,nN:nP] 
coeffs = mf.mo_coeff[n2c:,nN:nP]
sl = reduce(numpy.dot, (coeffl.conj().T,s,coeffl))
ss = reduce(numpy.dot, (coeffs.conj().T,t*(0.5/c)**2,coeffs))
pop = numpy.einsum('ij,ij->', sl+ss, rdm1)
print('Population : %s' % pop)

v = mol.intor_symmetric('int1e_nuc_spinor')
w = mol.intor_symmetric('int1e_spnucsp_spinor')
tls = 0.5*reduce(numpy.dot, (coeffl.conj().T,t,coeffs))
tsl = 0.5*reduce(numpy.dot, (coeffs.conj().T,t,coeffl))
vnl = reduce(numpy.dot, (coeffl.conj().T,v,coeffl))
wns = reduce(numpy.dot, (coeffs.conj().T,w*(0.5/c)**2-t*0.5,coeffs))
hcore = vnl + wns + tls + tsl
hcore = numpy.einsum('ij,ij->',hcore,rdm1)
print('Hcore energy : %s' % hcore)

coeffs = mf.mo_coeff[n2c:,nN:nP]*(0.5/c)
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

eri_mo = eri_llll + eri_ssss + eri_llss + eri_ssll
rdm2_j = einsum('ij,kl->ijkl', rdm1, rdm1) 
rdm2_xc = -einsum('il,kj->ijkl', rdm1, rdm1)
bie1 = numpy.einsum('ijkl,jilk->', eri_mo, rdm2_j)*0.5 
bie2 = numpy.einsum('ijkl,jilk->', eri_mo, rdm2_xc)*0.5
print('J energy : %s' % bie1)
print('XC energy : %s' % bie2)
print('EE energy : %s' % (bie1+bie2))
etot = enuc + hcore + bie1 + bie2
print('Total energy : %s' % etot)

