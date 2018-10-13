#!/usr/bin/env python

import numpy
from pyscf import scf, gto, qmmm, mcscf

mol = gto.Mole()
mol.atom = ''' O                  0.00000000    0.00000000   -0.11081188
               H                 -0.00000000   -0.84695236    0.59109389
               H                 -0.00000000    0.89830571    0.52404783 '''
mol.basis = 'cc-pvdz'
mol.verbose = 4
mol.build()

coords = [(0.5,0.6,0.8)]
charges = [-0.5]
mf = qmmm.itrf.mm_charge(scf.RHF(mol), coords, charges)
mf.kernel()

mc = mcscf.CASSCF(mf, 2, 2)
mc.kernel()

