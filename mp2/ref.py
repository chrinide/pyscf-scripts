#!/usr/bin/env python

import numpy
from pyscf import gto, scf, lib, ao2mo, mp
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
mol.symmetry = 0
mol.verbose = 4
mol.build()

mf = scf.RHF(mol)
mf.kernel()

pt2 = mp.MP2(mf)
pt2.frozen = 1
pt2.kernel()

