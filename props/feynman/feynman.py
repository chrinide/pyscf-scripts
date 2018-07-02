#!/usr/bin/env python

import numpy, sys
from pyscf import gto, scf

mol = gto.Mole()
mol.basis = 'cc-pcvtz'
mol.atom = '''
O     -0.000000000000   0.000000000000   0.213199727666
H      1.419923197638   0.000000000000  -0.852798910663
H     -1.419923197638   0.000000000000  -0.852798910663
'''
mol.verbose = 4
mol.unit = 'B'
mol.symmetry = 1
mol.build()

mf = scf.RHF(mol)
mf.kernel()
dm = mf.make_rdm1()

charges = mol.atom_charges()
coords = mol.atom_coords()
for i in range(mol.natm):
    q1 = charges[i]
    r1 = coords[i]
    mol.set_rinv_origin(r1)
    fen  = q1*numpy.einsum('xij,ji->x',mol.intor('int1e_hellfey', comp=3), dm)
    fnn = numpy.zeros(3)                        
    for j in range(mol.natm):
        if (i!=j):
            q2 = charges[j]
            r2 = coords[j]
            d3x = (r2[0]-r1[0])**2
            d3y = (r2[1]-r1[1])**2
            d3z = (r2[2]-r1[2])**2
            d3 = d3x+d3y+d3z
            d3 = numpy.power(d3,1.5)
            fnn[0] += q2*q1*(r1[0]-r2[0])/d3
            fnn[1] += q2*q1*(r1[1]-r2[1])/d3
            fnn[2] += q2*q1*(r1[2]-r2[2])/d3
    fnn /= 2.0
    print('Forces for atom %s' % i)
    print('E-N Forces %s' % fen)
    print('N-N Forces %s' % fnn)
    print('Total force on aton %s' % (fen+fnn))

