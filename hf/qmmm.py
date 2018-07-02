#!/usr/bin/python

from pyscf import gto, scf
from pyscf.qmmm import itrf

mol = gto.Mole()
mol.atom = ''' O                  0.00000000    0.00000000   -0.11081188
               H                 -0.00000000   -0.84695236    0.59109389
               H                 -0.00000000    0.89830571    0.52404783 '''
mol.basis = 'cc-pvdz'
mol.verbose = 4
mol.build()

coords = [(0.5,0.6,0.8)]
charges = [-0.5]
mf = itrf.mm_charge(scf.RHF(mol), coords, charges)
mf = scf.RHF(mol)
ehf = mf.kernel()

