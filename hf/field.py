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

E = [0.0,0.0,-0.1]
mol.set_common_orig([0, 0, 0]) # The gauge origin for dipole integral
k = mol.intor('int1e_kin') 
ne = mol.intor('int1e_nuc_sph')
ef = numpy.einsum('x,xij->ij', E, mol.intor('int1e_r_sph', comp=3))
h = k + ne + ef

mf = scf.RHF(mol)
mf.get_hcore = lambda *args: h
mf.kernel()

