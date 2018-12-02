#!/usr/bin/env python

import numpy
from pyscf import gto, scf, ao2mo, mcscf

mol = gto.Mole()
mol.build(
    atom = 'H 0 0 0; F 0 0 1.1',  # in Angstrom
    basis = 'ccpvdz',
    verbose = 4,
)

h = mol.intor('cint1e_kin_sph') + mol.intor('cint1e_nuc_sph')  
E = [0.0,0.0,-0.01]
origin = ([0.0,0.0,0.0])
charges = mol.atom_charges()
coords  = mol.atom_coords()
mol.set_common_orig(origin) # The gauge origin for dipole integral
e = numpy.einsum('x,xij->ij', E, mol.intor('cint1e_r_sph', comp=3)) 
h1 = h + e

mf = scf.RHF(mol)
mf.get_hcore = lambda *args: h1
mf.kernel()
mc = mcscf.CASSCF(mf, 2, 2)
mc.kernel()

