#!/usr/bin/env python

import numpy
import scipy.linalg
import os
from pyscf import scf
from pyscf import gto
from pyscf import mcscf, dmrgscf
from pyscf import tools
from pyscf import ao2mo
from pyscf import symm
from pyscf.dmrgscf import DMRGCI
from pyscf.tools import molden
from pyscf.tools import dump_mat

from pyscf.dmrgscf import settings
settings.MPIPREFIX = '/usr/bin/mpirun -x OMP_NUM_THREADS=1 -x MKL_NUM_THREADS=1 -np 4'
settings.BLOCKSCRATCHDIR = '/scratch-ssd/jluis/n2_fci'

name = 'n2_fci'

mol = gto.Mole()
mol.basis = 'cc-pvdz'
mol.atom = open('n2.xyz').read()
mol.verbose = 4
mol.spin = 0
mol.symmetry = 1
mol.charge = 0
mol.build()

mf = scf.RHF(mol)
mf.conv_tol = 1e-12
mf.direct_scf = False
mf.level_shift = 0.1
mf.kernel()

from pyscf.tools import localizer
occ = mf.mo_coeff[:,mf.mo_occ>0]
vir = mf.mo_coeff[:,mf.mo_occ==0]
loc_occ = localizer.localizer(mol, occ, 'edmiston')
loc_occ.verbose = 10
new_occ = loc_occ.optimize()
loc_vir = localizer.localizer(mol, vir, 'boys')
loc_vir.verbose = 10
new_vir = loc_vir.optimize()
loc_mo = numpy.hstack([new_occ,new_vir])

mc = mcscf.CASCI(mf, mf.mo_coeff.shape[1], mol.nelectron)
mc.fcisolver = DMRGCI(mol)
mc.fcisolver.memory = 8
mc.fcisolver.num_thrds = 2
mc.fcisolver.maxIter = 150
mc.fcisolver.max_cycle = 150
mc.fcisolver.dmrg_switch_tol = 1e-8
mc.fcisolver.tol = 1e-8
mc.fcisolver.maxM = 3200
mc.fcisolver.scheduleSweeps = [    0,    4,    6,    8,   12,   16,   20,   30 ]
mc.fcisolver.scheduleMaxMs  = [  500,  700, 1000, 1400, 1800, 2400, 2800, 3200 ]
mc.fcisolver.scheduleTols   = [ 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-5, 1e-8, 1e-8 ]
mc.fcisolver.scheduleNoises = [ 1e-2, 1e-2, 1e-2, 1e-3, 1e-4, 1e-4,  0.0,  0.0 ]
mc.fcisolver.twodot_to_onedot = 60
mc.fcisolver.configFile = "block-"+name+".conf"
mc.fcisolver.outputFile = "block-"+name+".out"
mc.fcisolver.integralFile = "FCIDUMP-"+name
mc.fcisolver.nroots = 1
mc.kernel(loc_mo)
