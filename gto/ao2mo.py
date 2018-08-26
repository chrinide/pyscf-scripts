#!/usr/bin/env python

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

mf = scf.RHF(mol)
mf.kernel()

ncore = 1
nact = 8
nelec = 8 
nao, nmo = mf.mo_coeff.shape
c = mf.mo_coeff
eri_ao = ao2mo.restore(1, mf._eri, nmo)
eri_mo = einsum('sl,pqrs->pqrl', c, eri_ao)
eri_mo = einsum('rk,pqrl->pqkl', c, eri_mo)
eri_mo = einsum('qj,pqkl->pjkl', c, eri_mo)
eri_mo = einsum('pi,pjkl->ijkl', c, eri_mo)

eri_mo = eri_mo + eri_mo.transpose(1,0,2,3)
eri_mo = eri_mo + eri_mo.transpose(0,1,3,2)
eri_mo = eri_mo + eri_mo.transpose(2,3,0,1)
eri_mo /= 8

eri_ref = ao2mo.kernel(mf._eri, c, compact=False)
eri_ref = eri_ref.reshape((nmo,nmo,nmo,nmo))
lib.logger.info(mf, "ERI diff: %s" % (abs(eri_mo - eri_ref).max()))

h1 = reduce(numpy.dot, (mf.mo_coeff[:,:nmo].T, mf.get_hcore(), mf.mo_coeff[:,:nmo]))
# CAS Hamiltonian
# \tilde{t}_{pq} = t_{pq} + \sum_{i\in\textrm{ncore}} (2\langle pq \vert ii \rangle - 
#                                                       \langle pi \vert iq \rangle) 
# e_\text{core} = e_\text{nn} + 2\sum_{i\in\textrm{ncore}} t_{ii} + 
#                 \sum_{i,j\in\textrm{ncore}} (2 \langle ii \vert jj \rangle - 
#                                                \langle ij \vert ji \rangle)
ecore = mol.energy_nuc()
ecore += 2*numpy.trace(h1[:ncore, :ncore])
ecore += 2*numpy.einsum('iijj', eri_mo[:ncore, :ncore, :ncore, :ncore])
ecore -= numpy.einsum('ijij', eri_mo[:ncore, :ncore, :ncore, :ncore])

# Active space one-body integrals
norb = ncore + nact
h1e = h1[ncore:norb, ncore:norb].copy()
h1e += 2*numpy.einsum('acbb->ac', eri_mo[ncore:norb, ncore:norb, :ncore, :ncore])
h1e -= numpy.einsum('abbc->ac', eri_mo[ncore:norb, :ncore, :ncore, ncore:norb])

# Active space two-body integrals
h2e = eri_mo[ncore:norb, ncore:norb, ncore:norb, ncore:norb]
h2e = ao2mo.restore(8,h2e,nact)

e = fci.direct_spin1.kernel(h1e, h2e, nact, nelec, ecore=ecore, verbose=4)[0]
lib.logger.info(mf, 'Core energy: %s' % ecore)
lib.logger.info(mf, 'Total energy: %s' % e)

# Generating the active space Hamiltonian with get_h1eff and get_h2eff function
lib.logger.info(mf, '################')
lib.logger.info(mf, 'Reference values')
lib.logger.info(mf, '################')
mc = mcscf.CASSCF(mf, nact, nelec)
h1e_cas, ecore = mc.get_h1eff()
h2e_cas = mc.get_h2eff()
e = fci.direct_spin1.kernel(h1e_cas, h2e_cas, nact, nelec, ecore=ecore, verbose=4)[0]
lib.logger.info(mc, 'Core energy: %s' % ecore)
lib.logger.info(mc, 'Total energy: %s' % e)

