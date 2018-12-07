#!/usr/bin/env python

import sys
import numpy

from pyscf import gto, scf, mcscf, symm, lib
from pyscf.tools import wfn_format
from pyscf.lib import logger

import ci
import avas

# TODO: UPdate for excited states and ss and more

class AsFCISolver(lib.StreamObject):
    def __init__(self, mol):
        self.stdout = mol.stdout
        self.verbose = mol.verbose
        self.myci = None
        self.mol = mol
        self.sshift = 0.2
        self.ss = 0 
        self.conv_tol = 1e-6
        self.nroots = 1

    #def dump_flags(self, verbose=None):
    #    logger.info(self, 'Number of electrons: %s', self.mol.nelec)

    def kernel(self, h1, h2, norb, nelec, ci0=None, ecore=0, **kwargs):
        self.myci = ci.CI(self.mol)
        self.myci = ci.fix_spin(self.myci, ss=self.ss, shift=self.sshift)
        self.myci.nroots = self.nroots
        #self.myci.verbose = 0
        e, civec = self.myci.kernel(h1, h2, norb, nelec, verbose=0)
        return e[0]+ecore, civec[0]

    def make_rdm1(self, fake_ci, norb, nelec):
        dm1 = self.myci.make_rdm1(fake_ci, norb, nelec)
        return dm1

    def make_rdm12(self, fake_ci, norb, nelec):
        dm1, dm2 = self.myci.make_rdm12(fake_ci, norb, nelec)
        return dm1, dm2

    def spin_square(self, fake_ci, norb, nelec):
        return self.myci.spin_square(fake_ci, norb, nelec)

mol = gto.Mole()
mol.basis = 'aug-cc-pvdz'
mol.atom = '''
N  0.0000  0.0000  0.5488
N  0.0000  0.0000 -0.5488
    '''
mol.verbose = 4
mol.spin = 0
mol.charge = 0
mol.symmetry = 0
mol.build()

mf = scf.RHF(mol)
mf.kernel()

ncore = 2
aolst1 = ['N 2s']
aolst2 = ['N 2p']
aolst = aolst1 + aolst2
ncas, nelecas, mo = avas.kernel(mf, aolst, threshold_occ=0.1, threshold_vir=1e-2, minao='minao', ncore=ncore)

mc = mcscf.CASSCF(mf, ncas, nelecas)
mc.fcisolver = AsFCISolver(mol)
mc.max_cycle_macro = 250
mc.max_cycle_micro = 7
mc.kernel(mo)

mycas = mcscf.CASSCF(mf, ncas, nelecas)
mycas.fix_spin_(ss=0,shift=0.2)
mycas.kernel(mo)

