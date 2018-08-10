#!/usr/bin/env python

import numpy, sys, os
sys.path.append('../tools')
import utils
from pyscf import gto, scf, lib, ao2mo, ao2mo, cc
from pyscf.cc import ccsd_t
from pyscf.cc import ccsd_t_lambda_slow as ccsd_t_lambda
from pyscf.cc import ccsd_t_rdm_slow as ccsd_t_rdm
einsum = lib.einsum

name = 'f2_ccsd_fno'
mol = gto.Mole()
mol.basis = 'aug-cc-pvtz'
mol.atom = '''
F      0.000000      0.000000      0.018162
F      0.000000      0.000000      1.393738
'''
mol.charge = 0
mol.spin = 0
mol.symmetry = 0
mol.verbose = 4
mol.build()
#
mf = scf.RHF(mol)
mf.chkfile = name+'.chk'
ehf = mf.kernel()
#
ncore = 2
lib.logger.info(mf,"* Core orbitals: %d" % ncore)
nocc = mol.nelectron//2 - ncore
mo_core = mf.mo_coeff[:,:ncore]
mo_occ = mf.mo_coeff[:,ncore:ncore+nocc]
co = mo_occ
eo = mf.mo_energy[ncore:ncore+nocc]
nocc = len(eo)
#
cv,ev = utils.getfno(mf,ncore,thresh_vir=1e-4)
nvir = len(ev)
e_denom = 1.0/(eo.reshape(-1,1,1,1)-ev.reshape(-1,1,1)+eo.reshape(-1,1)-ev)
eri_mo = ao2mo.general(mf._eri, (co,cv,co,cv), compact=False)
eri_mo = eri_mo.reshape(nocc,nvir,nocc,nvir)
lib.logger.info(mf,"* Virtual orbitals: %d" % (len(ev)))
#
rdm2_mp2 = numpy.zeros((nocc,nvir,nocc,nvir))
rdm2_mp2 = 2.0*einsum('iajb,iajb->iajb', eri_mo, e_denom)
rdm2_mp2 -= einsum('ibja,iajb->iajb', eri_mo, e_denom)
e_mp2 = numpy.einsum('iajb,iajb->', eri_mo, rdm2_mp2, optimize=True)
lib.logger.info(mf,"!*** E(MP2): %12.8f" % e_mp2)
lib.logger.info(mf,"!**** E(HF+MP2): %12.8f" % (e_mp2+ehf))
#
mo_vir = cv
coeff = numpy.hstack([mo_core,mo_occ,mo_vir])
nao, nmo = coeff.shape
nocc = mol.nelectron//2
occ = numpy.zeros(nmo)
for i in range(nocc):
    occ[i] = 2.0
#
mycc = cc.CCSD(mf, mo_coeff=coeff, mo_occ=occ)
mycc.diis_space = 10
mycc.frozen = ncore
mycc.conv_tol = 1e-6
mycc.conv_tol_normt = 1e-6
mycc.max_cycle = 150
ecc, t1, t2 = mycc.kernel()
nao, nmo = coeff.shape
eris = mycc.ao2mo()
e3 = ccsd_t.kernel(mycc, eris, t1, t2)
lib.logger.info(mycc,"* CCSD(T) energy : %12.6f" % (ehf+ecc+e3))
l1, l2 = ccsd_t_lambda.kernel(mycc, eris, t1, t2)[1:]
rdm1 = ccsd_t_rdm.make_rdm1(mycc, t1, t2, l1, l2, eris=eris)
rdm2 = ccsd_t_rdm.make_rdm2(mycc, t1, t2, l1, l2, eris=eris)
#
eri_mo = ao2mo.kernel(mf._eri, coeff[:,:nmo], compact=False)
eri_mo = eri_mo.reshape(nmo,nmo,nmo,nmo)
h1 = reduce(numpy.dot, (coeff[:,:nmo].T, mf.get_hcore(), coeff[:,:nmo]))
ecc =(numpy.einsum('ij,ji->', h1, rdm1)
    + numpy.einsum('ijkl,ijkl->', eri_mo, rdm2)*.5 + mf.mol.energy_nuc())
lib.logger.info(mycc,"* Energy with 1/2-RDM : %.8f" % ecc)    
