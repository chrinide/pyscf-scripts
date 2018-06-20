#!/usr/bin/env python

import numpy
from pyscf import gto, scf, lib, ao2mo

mol = gto.Mole()
mol.basis = 'cc-pvdz'
mol.atom = '''
O  0.0000   0.0000   0.1173
H  0.0000   0.7572  -0.4692
H  0.0000  -0.7572  -0.4692
    '''
mol.verbose = 4
mol.spin = 0
mol.symmetry = 0
mol.charge = 0
mol.build()

mf = scf.RHF(mol)
ehf = mf.kernel()
nao = mol.nao_nr()
dm = mf.make_rdm1()

s = mol.intor('int1e_ovlp')
t = mol.intor('int1e_kin')
v = mol.intor('int1e_nuc')
eri_ao = ao2mo.restore(1,mf._eri,nao)
eri_ao = eri_ao.reshape(nao,nao,nao,nao)

enuc = mol.energy_nuc() 
ekin = numpy.einsum('ij,ji->',t,dm)
pop = numpy.einsum('ij,ji->',s,dm)
elnuce = numpy.einsum('ij,ji->',v,dm)
lib.logger.info(mf,'Population : %12.6f' % pop)
lib.logger.info(mf,'Kinetic energy : %12.6f' % ekin)
lib.logger.info(mf,'Nuclear Atraction energy : %12.6f' % elnuce)
lib.logger.info(mf,'Nuclear Repulsion energy : %12.6f' % enuc)
bie1 = numpy.einsum('ijkl,ij,kl->',eri_ao,dm,dm)*0.5 # J
bie2 = numpy.einsum('ijkl,il,jk->',eri_ao,dm,dm)*0.25 # XC
pairs1 = numpy.einsum('ij,kl,ij,kl->',dm,dm,s,s) # J
pairs2 = numpy.einsum('ij,kl,li,kj->',dm,dm,s,s)*0.5 # XC
pairs = (pairs1 - pairs2)
lib.logger.info(mf,'Coulomb Pairs : %12.6f' % (pairs1))
lib.logger.info(mf,'XC Pairs : %12.6f' % (pairs2))
lib.logger.info(mf,'Pairs : %12.6f' % pairs)
lib.logger.info(mf,'J energy : %12.6f' % bie1)
lib.logger.info(mf,'XC energy : %12.6f' % -bie2)
lib.logger.info(mf,'EE energy : %12.6f' % (bie1-bie2))
etot = enuc + ekin + elnuce + bie1 - bie2
lib.logger.info(mf,'Total energy : %12.6f' % etot)

