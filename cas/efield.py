#!/usr/bin/env python

import numpy
from pyscf import gto, scf, mcscf

mol = gto.Mole()
mol.build(
    atom = 'H 0 0 0; F 0 0 1.1',  # in Angstrom
    basis = 'ccpvdz',
    verbose = 4,
)

h = mol.intor_symmetric('int1e_kin') + \
    mol.intor_symmetric('int1e_nuc')  
E = [0.0,0.0,0.0001]
coords  = mol.atom_coords()
charges = mol.atom_charges()
charge_center = numpy.einsum('i,ix->x', charges, coords) / charges.sum()
with mol.with_common_orig(charge_center):
    ao_dip = mol.intor_symmetric('int1e_r', comp=3)
e = numpy.einsum('x,xij->ij', E, ao_dip)
h1 = h + e

mf = scf.RHF(mol)
mf.get_hcore = lambda *args: h1
mf.kernel()
mc = mcscf.CASSCF(mf, 2, 2)
mc.kernel()

