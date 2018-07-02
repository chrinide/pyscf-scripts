#!/usr/bin/env python

from pyscf import scf, gto

mol = gto.Mole()
mol.atom = '''
  F      0.0000  0.0000  0.0000
  F      0.0000  0.0000  1.0000
  GHOST  0.0000  0.0000  0.5000
'''
mol.basis = {'F':'cc-pvdz', 
             'GHOST':gto.basis.load('cc-pvdz', 'F')} 
mol.symmetry = 1
mol.verbose = 4
mol.charge = 0
mol.spin = 0
mol.build()

mf = scf.RHF(mol)
mf.conv_tol = 1e-8
mf.kernel()

