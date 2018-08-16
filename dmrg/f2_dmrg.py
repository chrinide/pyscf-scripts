#!/usr/bin/env python

import numpy
import scipy.linalg
import os
from pyscf import scf
from pyscf import gto
from pyscf import cc, mcscf, dmrgscf
from pyscf import tools
from pyscf import ao2mo
from pyscf import symm, lo
from pyscf.dmrgscf import DMRGCI
from pyscf.tools import molden

def add_inactive_space_to_rdm(mol, mo_coeff, one_pdm, two_pdm):
    '''If a CASSCF calculation has been done, the final RDMs will
    not contain the doubly occupied inactive orbitals. This function will add
    them and return the full density matrices.
    '''

    # Find number of inactive electrons by taking the number of electrons
    # as the trace of the 1RDM, and subtracting from the total number of
    # electrons
    ninact = (mol.nelectron - int(round(numpy.trace(one_pdm)))) / 2
    norb = mo_coeff.shape[1] 
    nsizerdm = one_pdm.shape[0]

    one_pdm_ = numpy.zeros( (norb, norb) )
    # Add the core first.
    for i in range(ninact):
        one_pdm_[i,i] = 2.0

    # Add the rest of the density matrix.
    one_pdm_[ninact:ninact+nsizerdm,ninact:ninact+nsizerdm] = one_pdm[:,:]

    two_pdm_ = numpy.zeros( (norb, norb, norb, norb) )
    
    # Add on frozen core contribution, assuming that the inactive orbitals are
    # doubly occupied.
    for i in range(ninact):
        for j in range(ninact):
            two_pdm_[i,i,j,j] +=  4.0
            two_pdm_[i,j,j,i] += -2.0

    # Inactve-Active elements.
    for p in range(ninact):
        for r in range(ninact,ninact+nsizerdm):
            for s in range(ninact,ninact+nsizerdm):
                two_pdm_[p,p,r,s] += 2.0*one_pdm_[r,s]
                two_pdm_[r,s,p,p] += 2.0*one_pdm_[r,s]
                two_pdm_[p,r,s,p] -= one_pdm_[r,s]
                two_pdm_[r,p,p,s] -= one_pdm_[r,s]

    # Add active space.
    two_pdm_[ninact:ninact+nsizerdm,ninact:ninact+nsizerdm, \
             ninact:ninact+nsizerdm,ninact:ninact+nsizerdm] = \
             two_pdm[:,:]

    return one_pdm_, two_pdm_

from pyscf.dmrgscf import settings
settings.MPIPREFIX = 'mpirun -x OMP_NUM_THREADS=2 -x MKL_NUM_THREADS=1 -np 12'
settings.BLOCKSCRATCHDIR = '/scratch-ssd/jluis'

name = 'f2'

mol = gto.Mole()
mol.basis = 'aug-cc-pvtz'
mol.atom = '''
F  0.0000  0.0000  0.0000
F  0.0000  0.0000  1.4119
'''
mol.verbose = 4
mol.spin = 0
mol.symmetry = 1
mol.symmetry_subgroup = 'D2h'
mol.charge = 0
mol.build()

mf = scf.RHF(mol)
mf.conv_tol = 1e-12
mf.direct_scf = False
mf.level_shift = 0.1
mf.kernel()

##########################################################################################
#occ = mf.mo_coeff[:,mf.mo_occ>0]
#vir = mf.mo_coeff[:,mf.mo_occ==0]
#new_occ = lo.PM(mol, occ).kernel()
#new_vir = lo.PM(mol, vir).kernel()
#loc_mo = numpy.hstack([new_occ,new_vir])
mc = mcscf.CASSCF(mf, 14, 14)
mc.fcisolver.tol = 1e-8
mc.fcisolver.max_cycle = 250
mc.max_cycle_macro = 250
mc.max_cycle_micro = 7
mc.fcisolver.nroots = 1
mc.kernel()
loc_mo = mc.mo_coeff
##########################################################################################
nmo = mf.mo_coeff.shape[1]
ncore = 2
norb = nmo - ncore
nelec = mol.nelectron - 2*ncore
mc = mcscf.CASCI(mf, norb, nelec)
mc.fcisolver = DMRGCI(mol)
mc.fcisolver.maxIter = 150
mc.fcisolver.max_cycle = 150
mc.fcisolver.dmrg_switch_tol = 1e-8
mc.fcisolver.tol = 1e-8
mc.fcisolver.maxM = 3400
mc.fcisolver.memory = 12  # in GB
mc.fcisolver.num_thrds = 2
mc.fcisolver.scheduleSweeps = [    0,    5,   10,   15,   25,   35,   45 ]
mc.fcisolver.scheduleMaxMs  = [  500, 1000, 1400, 1800, 2200, 2600, 3400 ]
mc.fcisolver.scheduleTols   = [ 1e-4, 1e-4, 1e-4, 1e-4, 1e-5, 1e-8, 1e-8 ]
mc.fcisolver.scheduleNoises = [ 2e-1, 2e-1, 1e-1, 1e-1, 1e-2, 1e-3,  0.0 ]
mc.fcisolver.twodot_to_onedot = 65
mc.fcisolver.configFile = "block-"+name+".conf"
mc.fcisolver.outputFile = "block-"+name+".out"
mc.fcisolver.integralFile = "FCIDUMP-"+name
mc.fcisolver.nroots = 1
mc.kernel(loc_mo)
rdm1, rdm2 = mc.fcisolver.make_rdm12(mc.ci, norb, nelec)
rdm1, rdm2 = add_inactive_space_to_rdm(mol, mf.mo_coeff, rdm1, rdm2)
##########################################################################################
den_file = name+'.den'
fspt = open(den_file,'w')
fspt.write('CCIQA    ENERGY =      0.000000000000 THE VIRIAL(-V/T)=   2.00000000\n')
fspt.write('La matriz D es:\n')
for i in range(nmo):
    for j in range(nmo):
        fspt.write('%i %i %.16f\n' % ((i+1), (j+1), rdm1[i,j]))
fspt.write('La matriz d es:\n')
for i in range(nmo):
    for j in range(nmo):
        for k in range(nmo):
            for l in range(nmo):
                if (abs(rdm2[i,j,k,l]) > 1e-12):
                        fspt.write('%i %i %i %i %.16f\n' % ((i+1), (j+1), (k+1), (l+1), rdm2[i,j,k,l]))
fspt.close()                    
##########################################################################################
with open(name+'.mol', 'w') as f2:
    molden.header(mol, f2)
    molden.orbital_coeff(mol, f2, mc.mo_coeff, occ=mf.mo_occ)
cmd = '/home/jluis/bin/molden2aim ' + name
os.system(cmd)
cmd = 'cat ' + name + '.den ' + '>> ' + name + '.wfn'
os.system(cmd)
