#!/usr/bin/env python

import numpy
from pyscf import scf, gto, lib, lo, dft
from pyscf.tools import molden

mol = gto.Mole()
mol.verbose = 4
mol.atom = '''
O      0.000000      0.000000      0.118351
H      0.000000      0.761187     -0.469725
H      0.000000     -0.761187     -0.469725
'''
mol.basis = 'aug-cc-pvtz'
mol.charge = 1
mol.spin = 1
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
coeff = numpy.dot(s12inv,coeff)
diff = reduce(numpy.dot,(coeff.T,s,coeff)) - numpy.identity(nb)
lib.logger.info(mf,'Orthonormal %s' % numpy.linalg.norm(diff))
index = numpy.argsort(-nocc)
nocc = nocc[index]
coeff = coeff[:,index]
lib.logger.info(mf,'Natual occupancy %s' % nocc)

