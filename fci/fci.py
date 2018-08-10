#!/usr/bin/env python

import numpy
from functools import reduce
from pyscf import gto, scf, ao2mo, symm, fci, lib

mol = gto.M()
mol.atom ='''
N 0.00000000 0.00000000 1.03715729
N 0.00000000 0.00000000 -1.03715729
'''
mol.basis = 'sto-3g'
mol.symmetry = 1
mol.verbose = 4
mol.unit = 'bohr'
mol.build()

mf = scf.RHF(mol)
mf.run()

ncore = 2
core_idx = numpy.arange(ncore)
ncore = core_idx.size
cas_idx = numpy.arange(ncore, numpy.shape(mf.mo_coeff)[1])

e_core = mol.energy_nuc()
hcore = mf.get_hcore()
core_dm = numpy.dot(mf.mo_coeff[:, core_idx], mf.mo_coeff[:, core_idx].T)*2.0
e_core += numpy.einsum('ij,ji', core_dm, hcore)
corevhf = scf.hf.get_veff(mol, core_dm)
e_core += numpy.einsum('ij,ji', core_dm, corevhf)*0.5
h1e = reduce(numpy.dot, (mf.mo_coeff[:, cas_idx].T, hcore + corevhf, mf.mo_coeff[:, cas_idx]))
h2e = ao2mo.full(mf._eri, mf.mo_coeff[:, cas_idx])
nelec = mol.nelectron - ncore*2
norb = cas_idx.size

cisolver = fci.direct_spin1.FCI()
cisolver.max_cycle = 100
cisolver.conv_tol = 1e-8
cisolver.wfnsym = 'A1g'
cisolver.orbsym = symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb, mf.mo_coeff[:, cas_idx])
e, civec = cisolver.kernel(h1e, h2e, norb, nelec, ecore=e_core)
lib.logger.info(mf,"* Energy : %.8f" % e)    

