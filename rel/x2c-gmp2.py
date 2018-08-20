#!/usr/bin/env python

import numpy
from pyscf import gto, scf, x2c
from pyscf import ao2mo, lib, mp

mol = gto.Mole()
mol.verbose = 4 
mol.atom = '''
 HG            0.0000000000   0.0000000000   0.0946282025
 HG            0.0000000000  -0.0000000000   2.6985700299
'''
mol.basis = 'dzp-dk'
mol.charge = 2
mol.build()

mf = scf.RHF(mol)
enr = mf.kernel()

pt2 = mp.MP2(mf)
pt2.kernel()

mf = x2c.UHF(mol)
dm = mf.get_init_guess() + 0j
dm[0,:] += .1j
dm[:,0] -= .2j
dm[7,:] += .2j
dm[:,7] -= .1j
ex2c = mf.kernel()

#enuc = mol.energy_nuc() 
#print('Nuclear energy : %s' % enuc)
#print('**** Info on AO basis')
#dm = mf.make_rdm1()
eri_ao = mol.intor('int2e_spinor')
nao = eri_ao.shape[0]
#s = mol.intor('cint1e_ovlp_spinor')
#h = mf.get_hcore()
#hcore = numpy.einsum('ij,ji->', dm, h)
#pop = numpy.einsum('ij,ji->', dm, s)
#print('Population : %s' % pop)
#print('Hcore energy : %s' % hcore)

#bie1 = numpy.einsum('ijkl,ji,lk->',eri_ao,dm,dm)*0.5 # J
#bie2 = numpy.einsum('ijkl,li,jk->',eri_ao,dm,dm)*0.5 # XC
#pairs1 = numpy.einsum('ij,kl,ji,lk->',dm,dm,s,s) # J
#pairs2 = numpy.einsum('ij,kl,li,jk->',dm,dm,s,s)*0.5 # XC
#pairs = (pairs1 - pairs2)
#print('Coulomb Pairs : %s' % (pairs1))
#print('XC Pairs : %s' % (pairs2))
#print('Pairs : %s' % pairs)
#print('J energy : %s' % bie1)
#print('XC energy : %s' % -bie2)
#print('EE energy : %s' % (bie1-bie2))

#etot = enuc + hcore + bie1 - bie2
#print('Total energy : %s' % etot)

#print('**** Info on MO basis')
#coeff = mf.mo_coeff[:,mf.mo_occ>0]
#occ = mf.mo_occ[mf.mo_occ>0]
#rdm1 = numpy.zeros((len(occ),len(occ)),dtype=dm.dtype)
#rdm1 = numpy.diag(occ)
#h = reduce(numpy.dot, (coeff.conj().T,h,coeff))
#hcore = numpy.einsum('ij,ji->',h,rdm1)
#print('Hcore energy : %s' % hcore)
#nmo = mol.nelectron
#eri_mo = ao2mo.kernel(eri_ao, coeff, compact=False)
#eri_mo = eri_mo.reshape(nmo,nmo,nmo,nmo)
 
#rdm2_j = numpy.einsum('ij,kl->ijkl', rdm1, rdm1) 
#rdm2_xc = -numpy.einsum('il,kj->ijkl', rdm1, rdm1)
#rdm2 = rdm2_j + rdm2_xc
#bie1 = numpy.einsum('ijkl,jilk->', eri_mo, rdm2_j)*0.5 
#bie2 = numpy.einsum('ijkl,jilk->', eri_mo, rdm2_xc)*0.5
#print('J energy : %s' % bie1)
#print('XC energy : %s' % bie2)
#print('EE energy : %s' % (bie1+bie2))

#etot = enuc + hcore + bie1 + bie2
#print('Total energy : %s' % etot)

#################################################
print('**** Relativistic GMP2')
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
print("!*** E(MP2): %s" % e_mp2)
#print("!**** E(HF+MP2): %s" % (e_mp2+etot))

