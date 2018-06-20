#!/usr/bin/env python

import numpy, sys
from functools import reduce
from pyscf import gto, scf, cc, lib, ao2mo
from pyscf.cc import ccsd_t
from pyscf.cc import ccsd_t_lambda_slow as ccsd_t_lambda
from pyscf.cc import ccsd_t_rdm_slow as ccsd_t_rdm

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
mf = scf.newton(mf)
mf = scf.addons.remove_linear_dep_(mf)
mf.level_shift = 0.5
mf.conv_tol = 1e-8
mf.kernel()
dm = mf.make_rdm1()
mf.level_shift = 0.0
ehf = mf.kernel(dm)

ncore = 2
mcc = cc.CCSD(mf)
mcc.direct = 1
mcc.diis_space = 10
mcc.frozen = ncore
mcc.conv_tol = 1e-6
mcc.conv_tol_normt = 1e-6
mcc.max_cycle = 150
ecc, t1, t2 = mcc.kernel()

nao, nmo = mf.mo_coeff.shape
rdm1 = mcc.make_rdm1()
rdm2 = mcc.make_rdm2()
rdm2_j = numpy.einsum('ij,kl->ijkl', rdm1, rdm1)
rdm2_xc = rdm2 - rdm2_j
s = mol.intor('int1e_ovlp')
s = reduce(numpy.dot, (mf.mo_coeff[:,:nmo].T, s, mf.mo_coeff[:,:nmo]))
pairs1 = numpy.einsum('ijkl,ij,kl->',rdm2_j,s,s) # J
pairs2 = numpy.einsum('ijkl,ij,kl->',rdm2_xc,s,s) # XC
pairs = (pairs1 + pairs2)
lib.logger.info(mf,'Coulomb Pairs : %12.6f' % (pairs1))
lib.logger.info(mf,'XC Pairs : %12.6f' % (-pairs2))
lib.logger.info(mf,'Pairs : %12.6f' % pairs)

eris = mcc.ao2mo()
e3 = ccsd_t.kernel(mcc, eris, t1, t2)
lib.logger.info(mcc,"* CCSD(T) energy : %12.6f" % (ehf+ecc+e3))
l1, l2 = ccsd_t_lambda.kernel(mcc, eris, t1, t2)[1:]

rdm1 = ccsd_t_rdm.make_rdm1(mcc, t1, t2, l1, l2, eris=eris)
rdm2 = ccsd_t_rdm.make_rdm2(mcc, t1, t2, l1, l2, eris=eris)
pairs1 = numpy.einsum('ijkl,ij,kl->',rdm2_j,s,s) # J
pairs2 = numpy.einsum('ijkl,ij,kl->',rdm2_xc,s,s) # XC
pairs = (pairs1 + pairs2)
lib.logger.info(mf,'Coulomb Pairs : %12.6f' % (pairs1))
lib.logger.info(mf,'XC Pairs : %12.6f' % (-pairs2))
lib.logger.info(mf,'Pairs : %12.6f' % pairs)
