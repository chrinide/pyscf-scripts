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

mf = x2c.UHF(mol)
dm = mf.get_init_guess() + 0.1j
mf.kernel(dm)

enuc = mol.energy_nuc() 
print('Nuclear energy : %s' % enuc)
print('**** Info on AO basis')
dm = mf.make_rdm1()
eri_ao = mol.intor('int2e_spinor')
nao = eri_ao.shape[0]
s = mol.intor('cint1e_ovlp_spinor')
h = mf.get_hcore()
hcore = numpy.einsum('ij,ji->', dm, h)
pop = numpy.einsum('ij,ji->', dm, s)
print('Population : %s' % pop)
print('Hcore energy : %s' % hcore)

bie1 = numpy.einsum('ijkl,ji,lk->',eri_ao,dm,dm)*0.5 # J
bie2 = numpy.einsum('ijkl,li,jk->',eri_ao,dm,dm)*0.5 # XC
pairs1 = numpy.einsum('ij,kl,ji,lk->',dm,dm,s,s) # J
pairs2 = numpy.einsum('ij,kl,li,jk->',dm,dm,s,s)*0.5 # XC
pairs = (pairs1 - pairs2)
print('Coulomb Pairs : %s' % (pairs1))
print('XC Pairs : %s' % (pairs2))
print('Pairs : %s' % pairs)
print('J energy : %s' % bie1)
print('XC energy : %s' % -bie2)
print('EE energy : %s' % (bie1-bie2))

etot = enuc + hcore + bie1 - bie2
print('Total energy : %s' % etot)

print('**** Info on MO basis')
coeff = mf.mo_coeff[:,mf.mo_occ>0]
occ = mf.mo_occ[mf.mo_occ>0]
rdm1 = numpy.zeros((len(occ),len(occ)),dtype=dm.dtype)
rdm1 = numpy.diag(occ)
h = reduce(numpy.dot, (coeff.conj().T,h,coeff))
hcore = numpy.einsum('ij,ji->',h,rdm1)
print('Hcore energy : %s' % hcore)
nmo = mol.nelectron
eri_mo = ao2mo.kernel(eri_ao, coeff, compact=False)
eri_mo = eri_mo.reshape(nmo,nmo,nmo,nmo)

rdm2_j = numpy.einsum('ij,kl->ijkl', rdm1, rdm1) 
rdm2_xc = -numpy.einsum('il,kj->ijkl', rdm1, rdm1)
rdm2 = rdm2_j + rdm2_xc
bie1 = numpy.einsum('ijkl,jilk->', eri_mo, rdm2_j)*0.5 
bie2 = numpy.einsum('ijkl,jilk->', eri_mo, rdm2_xc)*0.5
print('J energy : %s' % bie1)
print('XC energy : %s' % bie2)
print('EE energy : %s' % (bie1+bie2))

etot = enuc + hcore + bie1 + bie2
print('Total energy : %s' % etot)

