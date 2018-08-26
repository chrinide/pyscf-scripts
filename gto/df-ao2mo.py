#!/usr/bin/env python

from pyscf import gto, scf, df
from pyscf.tools import molden

import numpy
from functools import reduce
from pyscf import gto, scf, lib, ao2mo, fci, mcscf
from pyscf.tools import molden
einsum = lib.einsum

mol = gto.Mole()
mol.basis = 'sto-6g'
mol.atom = '''
C  0.0000  0.0000  0.0000
H  0.6276  0.6276  0.6276
H  0.6276 -0.6276 -0.6276
H -0.6276  0.6276 -0.6276
H -0.6276 -0.6276  0.6276
'''
mol.charge = 0
mol.spin = 0
mol.symmetry = 1
mol.verbose = 4
mol.build()

mf = scf.density_fit(scf.RHF(mol))
mf.auxbasis = 'def2-svp-jkfit'
mf.kernel()

ncore = 1
nact = 8
nelec = 8 

c = mf.mo_coeff
nao, nmo = mf.mo_coeff.shape
naux = mf._cderi.shape[0]
dferi = numpy.empty((naux,nao,nao))
for i in range(naux):
    dferi[i] = lib.unpack_tril(mf._cderi[i])
eri_mo = einsum('rj,Qrs->Qjs', c, dferi)
eri_mo = einsum('sb,Qjs->Qjb', c, eri_mo)

h1 = reduce(numpy.dot, (mf.mo_coeff[:,:nmo].T, mf.get_hcore(), mf.mo_coeff[:,:nmo]))

ecore = mol.energy_nuc()
ecore += 2*numpy.trace(h1[:ncore, :ncore])
ecore += 2*numpy.einsum('Qii,Qjj->', eri_mo[:, :ncore, :ncore], eri_mo[:, :ncore, :ncore])
ecore -= numpy.einsum('Qij,Qji->', eri_mo[:, :ncore, :ncore], eri_mo[:, :ncore, :ncore])

# Active space one-body integrals
norb = ncore + nact
h1e = h1[ncore:norb, ncore:norb].copy()
h1e += 2*numpy.einsum('Qpq,Qii->pq', eri_mo[:, ncore:norb, ncore:norb], eri_mo[:, :ncore, :ncore])
h1e -= numpy.einsum('Qpi,Qiq->pq', eri_mo[:, ncore:norb, :ncore], eri_mo[:, :ncore, ncore:norb])

# Active space two-body integrals
h2e = eri_mo[:, ncore:norb, ncore:norb]
h2e = numpy.einsum('Qpq,Qrs->pqrs', h2e, h2e)
h2e = ao2mo.restore(8,h2e,nact)

e = fci.direct_spin1.kernel(h1e, h2e, nact, nelec, ecore=ecore, verbose=4)[0]
lib.logger.info(mf, 'Core energy: %s' % ecore)
lib.logger.info(mf, 'Total energy: %s' % e)

lib.logger.info(mf, '################')
lib.logger.info(mf, 'Reference values')
lib.logger.info(mf, '################')
mc = mcscf.DFCASSCF(mf, nact, nelec)
h1e_cas, ecore = mc.get_h1eff()
h2e_cas = mc.get_h2eff()
e = fci.direct_spin1.kernel(h1e_cas, h2e_cas, nact, nelec, ecore=ecore, verbose=4)[0]
lib.logger.info(mc, 'Core energy: %s' % ecore)
lib.logger.info(mc, 'Total energy: %s' % e)

