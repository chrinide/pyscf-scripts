#!/usr/bin/env python

from pyscf import gto, scf, mcscf

name = 'h2'

mol = gto.Mole()
mol.atom = '''
H  0.0000  0.0000  0.0000
H  0.0000  0.0000  2.0000
'''
mol.basis = 'aug-cc-pvdz'
mol.verbose = 4
mol.spin = 0
mol.symmetry = 1
mol.charge = 0
mol.build()

mf = scf.RHF(mol)
mf.max_cycle = 150
mf.chkfile = name+'.chk'
mf.level_shift = 0.5
mf.kernel()

ncas = mf.mo_coeff.shape[1]
nelecas = 2
mc = mcscf.CASCI(mf, ncas, nelecas)
mc.kernel()

