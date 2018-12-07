#!/usr/bin/env python

import numpy, sys
sys.path.append('../tools')
from pyscf import gto, scf, mcscf, solvent, dft
import avas

mol = gto.Mole()
mol.basis = 'aug-cc-pvdz'
mol.atom = '''
  F     0.0000  0.0000  0.0000
  Li    0.0000  0.0000  1.5639
    '''
mol.verbose = 4
mol.spin = 0
mol.charge = 0
mol.symmetry = 1
mol.build()

mf = scf.RHF(mol)
mf = solvent.ddCOSMO(mf)
mf.with_solvent.lebedev_order = 21
mf.with_solvent.lmax = 10 
mf.with_solvent.max_cycle = 50
mf.with_solvent.conv_tol = 1e-6
mf.with_solvent.grids.radi_method = dft.mura_knowles
mf.with_solvent.grids.becke_scheme = dft.stratmann
mf.with_solvent.grids.level = 4
mf.with_solvent.grids.prune = None
mf.kernel()

ncore = 2
aolst1 = ['F 2s']
aolst2 = ['F 2p']
aolst3 = ['Li 2s']
aolst4 = ['Li 2p']
aolst = aolst1 + aolst2 + aolst3 + aolst4
ncas, nelecas, mo = avas.kernel(mf, aolst, threshold_occ=0.1, threshold_vir=1e-5, minao='minao', ncore=ncore)

mc = mcscf.CASSCF(mf, ncas, nelecas)
mc = solvent.ddCOSMO(mc)
mc.with_solvent.lebedev_order = 21
mc.with_solvent.lmax = 10 
mc.with_solvent.max_cycle = 50
mc.with_solvent.conv_tol = 1e-6
mc.with_solvent.grids.radi_method = dft.mura_knowles
mc.with_solvent.grids.becke_scheme = dft.stratmann
mc.with_solvent.grids.level = 4
mc.with_solvent.grids.prune = None
mc.max_cycle_macro = 250
mc.max_cycle_micro = 7
mc.kernel(mo)

