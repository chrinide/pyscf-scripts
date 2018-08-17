#!/usr/bin/env python

import numpy
from pyscf import scf, gto, lo
from pyscf.tools import molden

mol = gto.Mole()
mol.basis = 'cc-pvdz'
mol.atom = '''
O
H 1 1.1
H 1 1.1 2 104
'''
mol.charge = 0
mol.spin = 0
mol.symmetry = 1
mol.verbose = 4
mol.build()

mf = scf.RHF(mol)
mf.kernel()

occ = mf.mo_coeff[:,mf.mo_occ>0]
vir = mf.mo_coeff[:,mf.mo_occ==0]

loc_occ = lo.ER(mol, occ).kernel()
loc_vir = lo.Boys(mol, vir).kernel()


