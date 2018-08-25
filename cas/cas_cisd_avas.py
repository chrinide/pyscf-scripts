#!/usr/bin/env python

import numpy, sys
from functools import reduce
sys.path.append('../tools')
from pyscf import gto, scf, ci, mcscf, ao2mo, lib
import avas

class AsFCISolver(object):
    def __init__(self):
        self.myci = None

    def kernel(self, h1, h2, norb, nelec, ci0=None, ecore=0, **kwargs):
        fakemol = gto.M(verbose=0)
        nelec = numpy.sum(nelec)
        fakemol.nelectron = nelec
        fake_hf = scf.RHF(fakemol)
        fake_hf._eri = ao2mo.restore(8, h2, norb)
        fake_hf.get_hcore = lambda *args: h1
        fake_hf.get_ovlp = lambda *args: numpy.eye(norb)
        fake_hf.kernel()
        self.myci = ci.CISD(fake_hf)
        self.myci.conv_tol = 1e-7
        self.myci.max_cycle = 150
        self.myci.max_space = 12
        self.myci.lindep = 1e-10
        self.myci.nroots = 1
        self.myci.level_shift = 0.2  # in precond
        e_corr, civec = self.myci.kernel()
        e = fake_hf.e_tot + e_corr
        return e+ecore, civec

    def make_rdm1(self, fake_ci, norb, nelec):
        mo = self.myci.mo_coeff
        civec = fake_ci      
        dm1 = self.myci.make_rdm1(civec)
        dm1 = reduce(numpy.dot, (mo, dm1, mo.T))
        return dm1
    
    def make_rdm12(self, fake_ci, norb, nelec):
        mo = self.myci.mo_coeff
        nmo = mo.shape[1]
        civec = fake_ci
        dm2 = self.myci.make_rdm2(civec)
        dm2 = numpy.dot(mo,dm2.reshape(nmo,-1))
        dm2 = numpy.dot(dm2.reshape(-1,nmo),mo.T)
        dm2 = dm2.reshape(nmo,nmo,nmo,nmo).transpose(2,3,0,1)
        dm2 = numpy.dot(mo,dm2.reshape(nmo,-1))
        dm2 = numpy.dot(dm2.reshape(-1,nmo),mo.T)
        dm2 = dm2.reshape(nmo,nmo,nmo,nmo)
        return self.make_rdm1(fake_ci, norb, nelec), dm2

    def spin_square(self, fake_ci, norb, nelec):
        return 0, 1

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
aolst3 = ['N 3s']
aolst4 = ['N 3p']
aolst = aolst1 + aolst2 + aolst3 + aolst4
ncas, nelecas, mo = avas.kernel(mf, aolst, threshold_occ=0.1, threshold_vir=0.01, minao='minao', ncore=ncore)

mc = mcscf.CASSCF(mf, ncas, nelecas)
mc.fcisolver = AsFCISolver()
mc.max_cycle_macro = 25
mc.max_cycle_micro = 7
mc.kernel(mo)

nmo = mc.ncore + mc.ncas
rdm1, rdm2 = mc.fcisolver.make_rdm12(mc.ci, mc.ncas, mc.nelecas)
rdm1, rdm2 = mcscf.addons._make_rdm12_on_mo(rdm1, rdm2, mc.ncore, mc.ncas, nmo)

eri_mo = ao2mo.kernel(mf._eri, mc.mo_coeff[:,:nmo], compact=False)
eri_mo = eri_mo.reshape(nmo,nmo,nmo,nmo)
h1 = reduce(numpy.dot, (mc.mo_coeff[:,:nmo].T, mf.get_hcore(), \
                        mc.mo_coeff[:,:nmo]))
e1 = numpy.einsum('ij,ij->', h1, rdm1)
e2 = numpy.einsum('ijkl,ijkl->', eri_mo, rdm2)*0.5 
etot = e1 + e2 + mf.mol.energy_nuc() 
lib.logger.info(mc,"* Energy with 1/2-RDM : %.8f" % etot)    

