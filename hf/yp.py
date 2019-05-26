#!/usr/bin/env python

from pyscf import gto, scf, ao2mo

mol = gto.Mole()
mol.incore_anyway = True
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

zeta = 0.05
mol.set_f12_zeta(zeta)
eri = mol.intor('int2e_yp')
#eri = mol.intor('int2e')

mf = scf.RHF(mol)
dm = mf.get_init_guess() 
n = dm.shape[1]
mf._eri = ao2mo.restore(8, eri, n)
mf.kernel()
