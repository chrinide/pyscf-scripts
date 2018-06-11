#!/usr/bin/env python

from pyscf import lib
from pyscf import gto, scf, dft
from pyscf.geomopt import berny_solver
from berny import Berny, geomlib, Logger, optimize as optimize_berny

name = 'c2h4_hf'

mol = gto.Mole()
mol.atom = '''
C  -0.471925  -0.471925  -1.859111
C   0.471925   0.471925  -1.859111
H  -0.872422  -0.872422  -0.936125
H   0.872422   0.872422  -0.936125
H  -0.870464  -0.870464  -2.783308
H   0.870464   0.870464  -2.783308
'''
mol.basis = 'aug-cc-pvdz'
mol.charge = 0
mol.spin = 0
mol.symmetry = 0
mol.verbose = 4
mol.build()

mf = scf.RHF(mol)
mf = scf.addons.remove_linear_dep_(mf)
mf.level_shift = 0.2
mf.conv_tol = 1e-8

mol = berny_solver.kernel(mf, maxsteps=50, assert_convergence=True)
xyzfile = name + '_opt.xyz'
fspt = open(xyzfile,'w')
coords = mol.atom_coords()*lib.param.BOHR
fspt.write('%d \n' % mol.natm)
fspt.write('%d %d\n' % (mol.charge, (mol.spin+1)))
for ia in range(mol.natm):
    symb = mol.atom_pure_symbol(ia)
    fspt.write('%s  %12.6f  %12.6f  %12.6f\n' % (symb, \
    coords[ia][0],coords[ia][1], coords[ia][2]))

