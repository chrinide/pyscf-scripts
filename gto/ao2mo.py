#!/usr/bin/env python

import numpy
from pyscf import gto, scf, lib, ao2mo
from pyscf.tools import molden
einsum = lib.einsum

mol = gto.Mole()
mol.basis = 'aug-cc-pvdz'
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
mf.level_shift = 0.5
mf = scf.addons.remove_linear_dep_(mf)
mf = scf.newton(mf)
mf.kernel()
dm = mf.make_rdm1()
mf.level_shift = 0.0
ehf = mf.kernel(dm)

nao, nmo = mf.mo_coeff.shape
c = mf.mo_coeff
eri_ao = ao2mo.restore(1, mf._eri, nmo)
eri_mo = einsum('rj,pqrs->pqjs', c, eri_ao)
eri_mo = einsum('pi,pqjs->iqjs', c, eri_mo)
eri_mo = einsum('sb,iqjs->iqjb', c, eri_mo)
eri_mo = einsum('qa,iqjb->iajb', c, eri_mo)

