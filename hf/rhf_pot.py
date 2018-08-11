#!/usr/bin/env python

import numpy
from pyscf import lib, ao2mo, gto, scf
einsum = lib.einsum

mol = gto.Mole()
mol.atom = '''
N 0 0 0 
N 0 0 1.1
            ''' 
mol.spin = 0
mol.symmetry = 1
mol.charge = 0
mol.verbose = 4
mol.basis = 'sto-3g'

mf = scf.RHF(mol)
mf.kernel()

dm = mf.make_rdm1()

s = mol.intor('int1e_ovlp')
k = mol.intor('int1e_kin')
v = mol.intor('int1e_nuc')
enuc = mol.energy_nuc() 
ekin = einsum('ij,ji->',k,dm)
pop = einsum('ij,ji->',s,dm)
lib.logger.info(mf,'Population : %12.6f' % pop)
lib.logger.info(mf,'Kinetic energy : %12.6f' % ekin)
lib.logger.info(mf,'Nuclear Repulsion energy : %12.6f' % enuc)

core = mf.get_hcore()
core = einsum('ij,ji->',core,dm)
lib.logger.info(mf,'1e-energy : %12.6f' % core)

nao = mol.nao_nr()
eri_ao = ao2mo.restore(1, mf._eri, nao)
eri_ao = eri_ao.reshape(nao,nao,nao,nao)
bie1 = einsum('ijkl,ij,kl',eri_ao,dm,dm)*0.5 # J
bie2 = numpy.einsum('ijkl,il,jk',eri_ao,dm,dm)*0.25 # XC
pairs1 = einsum('ij,kl,ij,kl->',dm,dm,s,s)*0.5 # J
pairs2 = numpy.einsum('ij,kl,li,kj->',dm,dm,s,s)*0.25 # XC
pairs = (pairs1 - pairs2)
lib.logger.info(mf,'Coulomb Pairs : %12.6f' % (pairs1))
lib.logger.info(mf,'XC Pairs : %12.6f' % (pairs2))
lib.logger.info(mf,'Pairs : %12.6f' % pairs)
lib.logger.info(mf,'J energy : %12.6f' % bie1)
lib.logger.info(mf,'XC energy : %12.6f' % -bie2)
lib.logger.info(mf,'EE energy : %12.6f' % (bie1-bie2))

vj, vk = mf.get_jk(dm)
lib.logger.info(mf,'J ref : %12.6f' % (einsum('ij,ji->', vj, dm)*0.5))
lib.logger.info(mf,'K ref : %12.6f' % (einsum('ij,ji->', vk, dm)*0.25))

etot = enuc + core + bie1 - bie2
lib.logger.info(mf,'Total energy : %12.6f' % etot)

hcore = mf.get_hcore()
e1 = numpy.einsum('ij,ji->',hcore,dm)
eri = eri_ao-0.5*eri_ao.transpose(0,3,2,1)
vhf = numpy.einsum('prqs,qs->pr',eri,dm)
e_coul = numpy.einsum('ij,ji->', vhf, dm)*0.5
lib.logger.info(mf,'Total energy from H_core+V_HF: %12.6f' % (e1+e_coul+enuc))

f = hcore + numpy.einsum('prqs,qs->pr',eri,dm)
e = 0.5*(hcore + f)
e = numpy.einsum('ij,ji->',e,dm)
lib.logger.info(mf,'Total energy from H_core+F: %12.6f' % (e+enuc))
e = 0.5*(numpy.trace(numpy.dot(dm,hcore+f)))
lib.logger.info(mf,'Total energy from Tr(H_core+F): %12.6f' % (e+enuc))

lib.logger.info(mf,'Fock operator in MO basis')
f = mf.mo_coeff.T.dot(f).dot(mf.mo_coeff)
lib.logger.info(mf,'%s' % f)
