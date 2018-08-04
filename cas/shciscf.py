#!/usr/bin/env python

import numpy, sys, struct, os
from functools import reduce
from pyscf import gto, scf, mcscf, dmrgscf, ao2mo, lib
from pyscf.shciscf import shci
from pyscf.tools import molden
import avas, rdm

mol = gto.Mole()
mol.basis = 'aug-cc-pvdz'
mol.atom = '''
C  0.0000  0.0000  0.0000
C  0.0000  0.0000  1.1888
    '''
mol.verbose = 4
mol.spin = 0
mol.charge = 0
mol.symmetry = 0
mol.build()

mf = scf.RHF(mol)
mf = scf.newton(mf)
mf = scf.addons.remove_linear_dep_(mf)
mf.level_shift = 0.5
mf.conv_tol = 1e-8
mf.kernel()
dm = mf.make_rdm1()
mf.level_shift = 0.0
ehf = mf.kernel(dm)

ncore = 2
aolst1 = ['C 2s']
aolst2 = ['C 2p']
aolst = aolst1 + aolst2
ncas, nelecas, mo = avas.kernel(mf, aolst, threshold_occ=0.1, threshold_vir=0.01, minao='ano', ncore=ncore)

mch = shci.SHCISCF(mf, ncas, nelecas)
mch.max_memory = 45000
mch.chkfile = name+'.chk'
mch.max_cycle_macro = 35
mch.max_cycle_micro = 7
mch.fcisolver.mpiprefix = '/opt/openmpi/1.8.4/bin/mpirun -np 12 ' 
mch.fcisolver.nPTiter = 0 # Turn off perturbative calc.
mch.fcisolver.sweep_iter = [0, 5]
mch.fcisolver.sweep_epsilon = [0.0005, 1e-4]
mch.fcisolver.dE = 1.e-6
mch.fcisolver.maxIter = 15
mch.fcisolver.num_thrds = 2
mch.fcisolver.useExtraSymm = True
mch.fcisolver.memory = 45000
mch.fcisolver.wfnsym = 'A1g'
#mo = lib.chkfile.load(name+'.chk', 'mcscf/mo_coeff')
mch.kernel(mo)

nmo = mch.ncore + mch.ncas
rdm1, rdm2 = mch.fcisolver.make_rdm12(mch.ci, mch.ncas, mch.nelecas)
rdm1, rdm2 = rdm.add_inactive_space_to_rdm(mol, nmo, rdm1, rdm2)

eri_mo = ao2mo.kernel(mf._eri, mch.mo_coeff[:,:nmo], compact=False)
eri_mo = eri_mo.reshape(nmo,nmo,nmo,nmo)
h1 = reduce(numpy.dot, (mch.mo_coeff[:,:nmo].T, mf.get_hcore(), mch.mo_coeff[:,:nmo]))
ecc =(numpy.einsum('ij,ij->', h1, rdm1)
    + numpy.einsum('ijkl,ijkl->', eri_mo, rdm2)*.5 + mf.mol.energy_nuc())
lib.logger.info(mch,"* Energy with 1/2-RDM : %.8f" % ecc)    

# Run a single SHCI iteration with perturbative correction.
mch.fcisolver.stochastic = False # Turns on deterministic PT calc.
mch.fcisolver.sampleN = 300
mch.fcisolver.epsilon2 = 1e-6
mch.fcisolver.targetError = 1.e-5
shci.writeSHCIConfFile(mch.fcisolver, [nelecas/2,nelecas/2] , False)
shci.executeSHCI(mch.fcisolver)

# Open and get the energy from the binary energy file shci.e.
file1 = open(os.path.join(mch.fcisolver.runtimeDir, "%s/shci.e"%(mch.fcisolver.prefix)), "rb")
format = ['d']*1
format = ''.join(format)
e_PT = struct.unpack(format, file1.read())
print "EPT:   ", e_PT
file1.close()

rdm1, rdm2 = mch.fcisolver.make_rdm12(mch.ci, mch.ncas, mch.nelecas)
rdm1, rdm2 = mcscf.addons._make_rdm12_on_mo(rdm1, rdm2, mch.ncore, mch.ncas, nmo)

eri_mo = ao2mo.kernel(mf._eri, mch.mo_coeff[:,:nmo], compact=False)
eri_mo = eri_mo.reshape(nmo,nmo,nmo,nmo)
h1 = reduce(numpy.dot, (mch.mo_coeff[:,:nmo].T, mf.get_hcore(), mch.mo_coeff[:,:nmo]))
ecc =(numpy.einsum('ij,ij->', h1, rdm1)
    + numpy.einsum('ijkl,ijkl->', eri_mo, rdm2)*.5 + mf.mol.energy_nuc())
lib.logger.info(mch,"* Energy with 1/2-RDM : %.8f" % ecc)    

# File cleanup
os.system("rm *.bkp")
os.system("rm *.txt")
os.system("rm shci.e")
os.system("rm *.dat")
os.system("rm FCIDUMP")
