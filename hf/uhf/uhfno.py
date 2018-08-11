#!/usr/bin/env python

import numpy, scipy
from functools import reduce
from pyscf import gto, scf, dft, lib

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

dm = mf.make_rdm1()
d = dm[0]+dm[1]
s = mol.intor('int1e_ovlp')
sl = scipy.linalg.sqrtm(s)

dt = reduce(numpy.dot, (sl,d,sl))
e, v = numpy.linalg.eig(dt)
natocc, natorb = numpy.linalg.eig(-dt)
natocc = natocc.real
natorb = natorb.real
for i, k in enumerate(numpy.argmax(abs(natorb), axis=0)):
    if natorb[k,i] < 0:
        natorb[:,i] *= -1
#natorb = numpy.dot(mc.mo_coeff[:,:nmo], natorb)
natocc = -natocc
tol = 1e-8
lib.logger.info(mf, '%s' % natocc[abs(natocc)>tol])

