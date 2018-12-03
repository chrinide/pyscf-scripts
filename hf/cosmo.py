#!/usr/bin/env python

from pyscf import gto, scf, dft
from pyscf import solvent

mol = gto.M(atom='''
C        0.000000    0.000000             -0.542500
O        0.000000    0.000000              0.677500
H        0.000000    0.9353074360871938   -1.082500
H        0.000000   -0.9353074360871938   -1.082500
            ''',
            verbose = 4)
mf = scf.RHF(mol)
mf = solvent.ddCOSMO(mf)
mf.with_solvent.lebedev_order = 21
mf.with_solvent.lmax = 10 
mf.with_solvent.eps = 78
mf.with_solvent.max_cycle = 50
mf.with_solvent.conv_tol = 1e-6
mf.with_solvent.grids.radi_method = dft.mura_knowles
mf.with_solvent.grids.becke_scheme = dft.stratmann
mf.with_solvent.grids.level = 4
mf.with_solvent.grids.prune = None
mf.kernel()

