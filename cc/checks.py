#!/usr/bin/env python

import numpy, sys
from functools import reduce
from pyscf import gto, scf, cc, lib, ao2mo
from pyscf.tools import molden

mol = gto.Mole()
mol.basis = 'aug-cc-pvdz'
mol.atom = '''
N  0.0000  0.0000  0.5488
N  0.0000  0.0000 -0.5488
'''
mol.verbose = 4
mol.spin = 0
mol.symmetry = 1
mol.charge = 0
mol.build()

mf = scf.RHF(mol)
mf.kernel()

ncore = 2
mcc = cc.CCSD(mf)
mcc.direct = 1
mcc.diis_space = 10
mcc.frozen = ncore
mcc.conv_tol = 1e-6
mcc.conv_tol_normt = 1e-6
mcc.max_cycle = 150
mcc.kernel()

nelec = mol.nelectron-ncore*2
t1norm = numpy.linalg.norm(mcc.t1)
t1norm = t1norm/numpy.sqrt(nelec)
lib.logger.info(mcc,"* T1 norm should be les than 0.02")
lib.logger.info(mcc,"* T1 norm : %12.6f" % t1norm)
t1norm = numpy.sqrt(numpy.linalg.norm(mcc.t1)**2 / nelec)
lib.logger.info(mcc,"* T1 norm : %12.6f" % t1norm)

f = lambda x: numpy.sqrt(numpy.sort(numpy.abs(x[0])))[-1]
d1norm_ij = f(numpy.linalg.eigh(numpy.einsum('ia,ja->ij',mcc.t1,mcc.t1)))
d1norm_ab = f(numpy.linalg.eigh(numpy.einsum('ia,ib->ab',mcc.t1,mcc.t1)))
d1norm = max(d1norm_ij, d1norm_ab)
lib.logger.info(mcc,"* D1 norm should be les than 0.05")
lib.logger.info(mcc,"* D1 norm : %12.6f" % d1norm)

d2norm_ij = f(numpy.linalg.eigh(numpy.einsum('ikab,jkab->ij',mcc.t2,mcc.t2)))
d2norm_ab = f(numpy.linalg.eigh(numpy.einsum('ijac,ijbc->ab',mcc.t2,mcc.t2)))
d2norm = max(d2norm_ij, d2norm_ab)
lib.logger.info(mcc,"* D2 norm : %12.6f" % d2norm)

