#!/usr/bin/env python

from pyscf import lib
from pyscf import gto, scf, dft
from pyscf.geomopt import berny_solver
from berny import Berny, geomlib, Logger, optimize as optimize_berny

name = 'c2h4'

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

mf = dft.RKS(mol).nuc_grad_method().as_scanner() 
mf.base = scf.addons.remove_linear_dep_(mf.base)
mf.base.level_shift = 0.2
mf.base.verbose = 4
mf.base.xc = 'pbe0'
mf.base.grids.level = 3
mf.base.grids.prune = dft.gen_grid.nwchem_prune
mf.base.conv_tol = 1e-8
mf.grid_response = True

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

