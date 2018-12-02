#!/usr/bin/env python

import numpy, sys
sys.path.append('../tools')
from pyscf import gto, scf, mcscf, solvent
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
mc.max_cycle_macro = 250
mc.max_cycle_micro = 7
mc.kernel(mo)

