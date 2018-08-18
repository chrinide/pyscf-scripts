#!/bin/bash            

import numpy
from pyscf import gto, scf, lib, ao2mo
from pyscf.tools import molden

name = 'h2o'

mol = gto.Mole()
mol.basis = 'cc-pvdz'
mol.atom = '''
O
H 1 1.1
H 1 1.1 2 104
'''
mol.charge = 0
mol.spin = 1
mol.charge = 1
mol.symmetry = 1
mol.verbose = 4
mol.build()

mf = scf.UHF(mol)
mf.kernel()

nao = mol.nao_nr()
dm = mf.make_rdm1()
da = dm[0]
db = dm[1]

s = mol.intor('int1e_ovlp')
t = mol.intor('int1e_kin')
v = mol.intor('int1e_nuc')
eri_ao = ao2mo.restore(1,mf._eri,nao)
eri_ao = eri_ao.reshape(nao,nao,nao,nao)

enuc = mol.energy_nuc() 
ekin = numpy.einsum('ij,ji->',t,da)
ekin += numpy.einsum('ij,ji->',t,db)
pop = numpy.einsum('ij,ji->',s,da)
pop += numpy.einsum('ij,ji->',s,db)
elnuce = numpy.einsum('ij,ji->',v,da)
elnuce += numpy.einsum('ij,ji->',v,db)
lib.logger.info(mf,'Population : %12.6f' % pop)
lib.logger.info(mf,'Kinetic energy : %12.6f' % ekin)
lib.logger.info(mf,'Nuclear Atraction energy : %12.6f' % elnuce)
lib.logger.info(mf,'Nuclear Repulsion energy : %12.6f' % enuc)

# fa = h + ja - ka + jb
# fb = h + jb - kb + ja
ja = numpy.einsum('pqrs,rs->pq', eri_ao, da)
jb = numpy.einsum('pqrs,rs->pq', eri_ao, db)
ka = numpy.einsum('prqs,rs->pq', eri_ao, da)
kb = numpy.einsum('prqs,rs->pq', eri_ao, db)
h = mf.get_hcore()
fa = h + ja - ka + jb
fb = h + jb - kb + ja
etot = 0.5*(numpy.einsum('pq,pq->', fa+h, da) + \
            numpy.einsum('pq,pq->', fb+h, db)) + enuc
lib.logger.info(mf,'Total energy : %12.6f' % etot)

