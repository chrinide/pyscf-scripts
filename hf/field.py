#!/usr/bin/env python

import numpy
from pyscf import gto, scf

mol = gto.Mole()
mol.atom = '''
N 0.0 0.0  0.55
N 0.0 0.0 -0.55
'''
mol.charge = 0
mol.spin = 0
mol.verbose = 4
mol.basis = 'cc-pVDZ'
mol.symmetry = 0
mol.build(dump_input=False, parse_arg=False)

k = mol.intor('int1e_kin') 
ne = mol.intor('int1e_nuc_sph')
E = [0.0001, 0, 0]
charges = mol.atom_charges()
coords  = mol.atom_coords()
charge_center = numpy.einsum('i,ix->x', charges, coords) / charges.sum()
with mol.with_common_orig(charge_center):
    ao_dip = mol.intor_symmetric('int1e_r', comp=3)
ef = numpy.einsum('x,xij->ij', E, ao_dip)
h = k + ne + ef

mf = scf.RHF(mol)
mf.get_hcore = lambda *args: h
mf.kernel()

