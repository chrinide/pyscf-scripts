#!/usr/bin/env python

import numpy
from pyscf import scf, gto, lib, lo, dft, ao2mo
from pyscf.tools import molden

mol = gto.Mole()
mol.basis = 'aug-cc-pvdz'
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

mf = dft.UKS(mol)
mf.xc = 'pbe0'
mf.grids.level = 4
mf.scf()

def sqrtm(s):
    e, v = numpy.linalg.eigh(s)
    return numpy.dot(v*numpy.sqrt(e), v.T.conj())
def lowdin(s):
    e, v = numpy.linalg.eigh(s)
    return numpy.dot(v/numpy.sqrt(e), v.T.conj())

mo_coeff = mf.mo_coeff
nb = mo_coeff[1].shape[1]
s = mol.intor_symmetric("cint1e_ovlp_sph")
dm = mf.make_rdm1()
h1e = mf.get_hcore()
e1 = numpy.einsum('ij,ji->', h1e, dm[0])
e1 += numpy.einsum('ij,ji->', h1e, dm[1])
lib.logger.info(mf,'H_core energy in AO basis %s' % e1)

# Natural orbitals
thresh = 1e-14
dm = 0.5*(dm[0]+dm[1])
s12 = sqrtm(s)
s12inv = lowdin(s)
dm = reduce(numpy.dot,(s12,dm,s12))
lib.logger.info(mf,'Idempotency of DM %s' % numpy.linalg.norm(dm.dot(dm)-dm))
nocc, coeff = numpy.linalg.eigh(dm)
nocc = 2*nocc
nocc[abs(nocc)<thresh] = 0.0
nocc[abs(nocc)>2.0-thresh] = 2.0
coeff = numpy.dot(s12inv,coeff)
dm = mf.make_rdm1(coeff, nocc)
lib.logger.info(mf,'Idempotency of DM %s' % numpy.linalg.norm(dm.dot(dm)-dm))
diff = reduce(numpy.dot,(coeff.T,s,coeff)) - numpy.identity(nb)
lib.logger.info(mf,'Orthonormal %s' % numpy.linalg.norm(diff))
index = numpy.argsort(-nocc)
nocc = nocc[index]
coeff = coeff[:,index]
lib.logger.info(mf,'Natural occupancy %s' % nocc[nocc>0])

# Energy in NO basis
nmo = nocc.shape[0]
rdm1 = numpy.zeros((nmo,nmo))
rdm1 = numpy.diag(nocc)
h1e = mf.get_hcore()
h1e = reduce(numpy.dot,(coeff.T,h1e,coeff))
e1 = numpy.einsum('ij,ji', h1e, rdm1)
lib.logger.info(mf,'H_core energy in NO basis %s' % e1)

eri_ao = ao2mo.restore(1,mf._eri,nmo)
eri_ao = eri_ao.reshape((nmo,nmo,nmo,nmo))
# Now eri_ao is the ERI tensor in Mulliken (11,22)
# To be consistent with promolden 
# 1) Obtain 2-RDM
# 2) Transform eri_ao -> eri_mo ao2mo.kernel(eri_ao,matrix)
# 3) Contract e2 = numpy.einsum('pqrs,pqrs->', eri_mo, rdm2)*0.5
# 4) energy = e1 + e2 + mol.energy_nuc()
#lib.logger.info(mf,'Total energy in NO basis: %s' % energy)
