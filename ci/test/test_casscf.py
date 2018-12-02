#!/usr/bin/env python

import numpy, sys
sys.path.append('../tools')
from pyscf import gto, scf, mcscf, symm, fci, lib, hci, ao2mo
from pyscf.tools import molden
from pyscf.lib import logger
import avas, wfn_format

# TODO: UPdate for excited states and ss

class AsFCISolver(lib.StreamObject):
    def __init__(self, mol):
        self.stdout = mol.stdout
        self.verbose = mol.verbose
        self.myci = None
        self.mol = mol
        self.shift = 0.2
        self.ss = 0.0000 
        self.ci_coeff_cutoff = .5e-3
        self.select_cutoff = .5e-3
        self.conv_tol = 1e-9
        self.conv_ndet_tol = 0.001
        self.nroots = 1
        self.max_iter = 100

    def dump_flags(self, verbose=None):
        logger.info(self, '\nStarting heat-bath CI algorithm...')
        logger.info(self, 'Selection threshold:                  %8.5e',    self.select_cutoff)
        logger.info(self, 'CI coefficient cutoff:                %8.5e',    self.ci_coeff_cutoff)
        logger.info(self, 'Number of determinants tolerance:     %8.5e',    self.conv_ndet_tol)
        logger.info(self, 'Number of electrons:                  %s',       self.mol.nelec)
        logger.info(self, 'Number of roots:                      %3d',      self.nroots)

    def kernel(self, h1, h2, norb, nelec, ci0=None, ecore=0, **kwargs):
        self.myci = hci.SelectedCI(self.mol)
        self.myci = hci.fix_spin(self.myci, ss=self.ss, shift=self.shift)
        self.myci.select_cutoff = self.select_cutoff
        self.myci.ci_coeff_cutoff = self.ci_coeff_cutoff
        self.myci.conv_ndet_tol = self.conv_ndet_tol
        self.myci.nroots = self.nroots
        self.myci.max_iter = self.max_iter
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

name = 'n2'

mol = gto.Mole()
mol.basis = 'aug-cc-pvdz'
mol.atom = '''
N  0.0000  0.0000  0.5488
N  0.0000  0.0000 -0.5488
    '''
mol.verbose = 4
mol.spin = 0
mol.charge = 0
mol.symmetry = 'c1'
mol.build()

mf = scf.RHF(mol)
mf = scf.newton(mf)
mf = scf.addons.remove_linear_dep_(mf)
mf.chkfile = name+'.chk'
#mf.__dict__.update(lib.chkfile.load(name+'.chk', 'scf'))
mf.level_shift = 0.5
mf.conv_tol = 1e-8
mf.kernel()
dm = mf.make_rdm1()
mf.level_shift = 0.0
ehf = mf.kernel(dm)

ncore = 2
aolst1 = ['N 2s']
aolst2 = ['N 2p']
aolst = aolst1 + aolst2
ncas, nelecas, mo = avas.kernel(mf, aolst, threshold_occ=0.1, threshold_vir=0.01, minao='minao', ncore=ncore)

mc = mcscf.CASSCF(mf, ncas, nelecas)
mc.fcisolver = AsFCISolver(mol)
mc.max_cycle_macro = 250
mc.max_cycle_micro = 7
mc.chkfile = name+'.chk'
#mc.__dict__.update(scf.chkfile.load(name+'.chk', 'mcscf'))
#mo = lib.chkfile.load(name+'.chk', 'mcscf/mo_coeff')
mc.kernel(mo)

nmo = mc.ncore + mc.ncas
rdm1, rdm2 = mc.fcisolver.make_rdm12(mc.ci, mc.ncas, mc.nelecas) 
rdm1, rdm2 = mcscf.addons._make_rdm12_on_mo(rdm1, rdm2, mc.ncore, mc.ncas, nmo)

orbsym = symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb, mc.mo_coeff[:,:nmo])
natocc, natorb = symm.eigh(-rdm1, orbsym)
for i, k in enumerate(numpy.argmax(abs(natorb), axis=0)):
    if natorb[k,i] < 0:
        natorb[:,i] *= -1
natorb = numpy.dot(mc.mo_coeff[:,:nmo], natorb)
natocc = -natocc

wfn_file = name + '.wfn'
with open(wfn_file, 'w') as f2:
    wfn_format.write_mo(f2, mol, natorb, mo_occ=natocc)
    wfn_format.write_coeff(f2, mol, mc.mo_coeff[:,:nmo])
    wfn_format.write_hci(f2, mc.ci, mc.ncas, mc.nelecas, root=0, ncore=mc.ncore)

