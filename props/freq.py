#!/usr/bin/env python

import numpy
from pyscf import gto, scf, dft
from pyscf.prop.freq import rks
from pyscf.data import elements

mol = gto.Mole()
mol.atom = '''
O      0.000000      0.000000      0.125215
H      0.000000      0.754109     -0.473158
H      0.000000     -0.754109     -0.473158
'''
mol.basis = 'def2-svp'
mol.charge = 0
mol.spin = 0
mol.symmetry = 1
mol.verbose = 4
mol.build()

mf = dft.RKS(mol)
mf.xc = 'pbe0'
mf.grids.level = 4
mf.grids.prune = dft.gen_grid.nwchem_prune
mf.grids.radi_method = dft.mura_knowles
mf.grids.becke_scheme = dft.stratmann
mf.grids.prune = None
mf.conv_tol = 1e-8
mf.kernel()

w, modes = rks.Freq(mf).kernel()

atom_charges = mol.atom_charges()
atmlst = numpy.where(atom_charges != 0)[0]  # Exclude ghost atoms
masses = numpy.array([elements.MASSES[atom_charges[i]] for i in atmlst])

M = numpy.diag(1.0/numpy.sqrt(numpy.repeat(masses, 3))) # Matrix for mass weighting
# Un-mass-weight the normal modes
t = modes.reshape(mol.natm*3,mol.natm*3)
t = M.dot(t)
t = t.reshape(-1,mol.natm,3)

for i in range(3*mol.natm):
    print (" mode %d " % i)
    for j in range(mol.natm):
        print ("atom %d %s" % (j,modes[i,j,:]))
    print ("Follow t")
    for j in range(mol.natm):
        print ("atom %d %s" % (j,t[i,j,:]))
