#!/usr/bin/env python

# TODO: update hci.c with cuttof when c0[i] is less than or
# based in ERI screaning
# TODO: add density fitting
# TODO: add noise on ERI
# TODO: improve hdiag

# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#         Alexander Sokolov <alexander.y.sokolov@gmail.com>
#

import numpy
import time
import ctypes

import cistring
import rdm

from pyscf import lib
from pyscf import ao2mo
from pyscf.lib import logger
from pyscf.fci import direct_spin1

libhci = lib.load_library('libhci')
einsum = lib.einsum

def contract_2e(h1_h2, civec, norb, nelec, hdiag=None, **kwargs):
    h1, eri = h1_h2
    strs = civec._strs
    ndet = len(strs)
    if hdiag is None:
        hdiag = make_hdiag(h1, eri, strs, norb, nelec)
    ci1 = numpy.zeros_like(civec)

    h1 = numpy.asarray(h1, order='C')
    eri = numpy.asarray(eri, order='C')
    strs = numpy.asarray(strs, order='C')
    civec = numpy.asarray(civec, order='C')
    hdiag = numpy.asarray(hdiag, order='C')
    ci1 = numpy.asarray(ci1, order='C')

    libhci.contract_h_c(h1.ctypes.data_as(ctypes.c_void_p), 
                        eri.ctypes.data_as(ctypes.c_void_p), 
                        ctypes.c_int(norb), 
                        ctypes.c_int(nelec[0]), 
                        ctypes.c_int(nelec[1]), 
                        strs.ctypes.data_as(ctypes.c_void_p), 
                        civec.ctypes.data_as(ctypes.c_void_p), 
                        hdiag.ctypes.data_as(ctypes.c_void_p), 
                        ctypes.c_ulonglong(ndet), 
                        ci1.ctypes.data_as(ctypes.c_void_p))

    return ci1

def spin_square(civec, norb, nelec):
    ss = numpy.dot(civec.T, contract_ss(civec, norb, nelec))
    s = numpy.sqrt(ss+.25) - .5
    multip = s*2+1
    return ss, multip

def contract_ss(civec, norb, nelec):
    strs = civec._strs
    ndet = len(strs)
    ci1 = numpy.zeros_like(civec)

    strs = numpy.asarray(strs, order='C')
    civec = numpy.asarray(civec, order='C')
    ci1 = numpy.asarray(ci1, order='C')

    libhci.contract_ss_c(ctypes.c_int(norb), 
                        ctypes.c_int(nelec[0]), 
                        ctypes.c_int(nelec[1]), 
                        strs.ctypes.data_as(ctypes.c_void_p), 
                        civec.ctypes.data_as(ctypes.c_void_p), 
                        ctypes.c_ulonglong(ndet), 
                        ci1.ctypes.data_as(ctypes.c_void_p))

    return ci1

def make_hdiag(h1e, eri, strs, norb, nelec):
    eri = ao2mo.restore(1, eri, norb)
    diagj = einsum('iijj->ij',eri)
    diagk = einsum('ijji->ij',eri)

    ndet = len(strs)
    hdiag = numpy.zeros(ndet)
    for idet, (stra, strb) in enumerate(strs.reshape(ndet,2,-1)):
        aocc = cistring.str2orblst(stra, norb)[0]
        bocc = cistring.str2orblst(strb, norb)[0]
        e1 = h1e[aocc,aocc].sum() + h1e[bocc,bocc].sum()
        e2 = diagj[aocc][:,aocc].sum() + diagj[aocc][:,bocc].sum() \
           + diagj[bocc][:,aocc].sum() + diagj[bocc][:,bocc].sum() \
           - diagk[aocc][:,aocc].sum() - diagk[bocc][:,bocc].sum()
        hdiag[idet] = e1 + e2*.5
    return hdiag

def kernel(myci, h1e, eri, norb, nelec, ci0=None,
           tol=None, lindep=None, max_cycle=None, max_space=None,
           nroots=None, davidson_only=None, 
           max_memory=None, verbose=None, ecore=0, **kwargs):

    if tol is None: tol = myci.conv_tol
    if lindep is None: lindep = myci.lindep
    if max_cycle is None: max_cycle = myci.max_cycle
    if max_space is None: max_space = myci.max_space
    if max_memory is None: max_memory = myci.max_memory
    if nroots is None: nroots = myci.nroots
    if verbose is None: verbose = logger.Logger(myci.stdout, myci.verbose)
    tol_residual = None #getattr(fci, 'conv_tol_residual', None)
    myci.dump_flags()

    nelec = direct_spin1._unpack_nelec(nelec, myci.spin)
    eri = ao2mo.restore(1, eri, norb)
    eri = eri.ravel()

    logger.info(myci, 'CI in the selected space')
    # Initial guess
    if ci0 is None:
        t_start = time.time()
        if (myci.model == 'fci'):
            dets = cistring.gen_full_space(range(norb), nelec) 
            ndets = dets.shape[0]
            if (myci.noise):
                numpy.random.seed()
                ci0 = [cistring.as_SCIvector(numpy.random.uniform(low=1e-3,high=1e-2,size=ndets), dets)]
                ci0[0][0] = 0.99
                ci0[0] *= 1./numpy.linalg.norm(ci0[0])
            else:
                ci0 = [cistring.as_SCIvector(numpy.zeros(ndets), dets)]
                ci0[0][0] = 1.0
        else:
            raise RuntimeError('''Unknown CI model''')
        t_current = time.time() - t_start
        #logger.info(myci, 'Initial CI vector = %s', ci0[0][:])
        logger.info(myci, 'Timing for generating strings: %10.3f', t_current)
    else:
        assert(nroots == len(ci0))

    def hop(c):
        hc = myci.contract_2e((h1e, eri), cistring.as_SCIvector(c, ci_strs), norb, nelec, hdiag)
        return hc.ravel()
    precond = lambda x, e, *args: x/(hdiag-e+myci.level_shift)

    ci_strs = ci0[0]._strs
    logger.info(myci, 'Number of CI configurations: %d', ci_strs.shape[0])
    t_start = time.time()
    hdiag = myci.make_hdiag(h1e, eri, ci_strs, norb, nelec)
    t_current = time.time() - t_start
    logger.info(myci, 'Timing for diagonal hamiltonian: %10.3f', t_current)

    t_start = time.time()
    with lib.with_omp_threads(myci.threads):
        #e, c = myci.eig(hop, ci0, precond, tol=tol, lindep=lindep,
        #                max_cycle=max_cycle, max_space=max_space, nroots=nroots,
        #                max_memory=max_memory, verbose=verbose, **kwargs)
        #
        e, c = myci.eig(hop, ci0, precond, tol=tol, lindep=lindep,
                       max_cycle=max_cycle, max_space=max_space, nroots=nroots,
                       max_memory=max_memory, verbose=verbose, follow_state=True,
                       tol_residual=tol_residual, **kwargs)
    t_current = time.time() - t_start
    logger.info(myci, 'Timing for solving the eigenvalue problem: %10.3f', t_current)
    if not isinstance(c, (tuple, list)):
        c = [c]
        e = [e]
    logger.info(myci, 'CI  E = %s', numpy.array(e)+ecore)
    #myci.eci = e+ecore
    #myci.ci = e
    #for i in range(nroots):
    #    norm = numpy.einsum('i,i->', ci0[i][:], ci0[i][:])
    #    s = myci.spin_square(ci0[i], norb, nelec)
    #    logger.info(myci, 'E(CI) = %10.5f, CI E = %10.5f, Norm = %1.3f, Spin = %s'\
    #    % (numpy.array(e[i]), numpy.array(e[i])+ecore, norm, s))
    #    #logger.info(myci, 'CI E = %s', numpy.array(e)+ecore)
    #    #logger.info(mc,"* Norm info for state %d : %s" % (i,norm))    
    #    #logger.info(mc,"* Spin info for state %d : %s" % (i,s))    

    return (numpy.array(e)+ecore), [cistring.as_SCIvector(ci, ci_strs) for ci in c]

def fix_spin(myci, shift=.2, ss=None, **kwargs):
    r'''If Selected CI solver cannot stick on spin eigenfunction, modify the solver by
    adding a shift on spin square operator
    .. math::
        (H + shift*S^2) |\Psi\rangle = E |\Psi\rangle
    Args:
        myci : An instance of :class:`SelectedCI`
    Kwargs:
        shift : float
            Level shift for states which have different spin
        ss : number
            S^2 expection value == s*(s+1)
    Returns
            A modified Selected CI object based on myci.
    '''
    if 'ss_value' in kwargs:
        sys.stderr.write('fix_spin_: kwarg "ss_value" will be removed in future release. '
                         'It was replaced by "ss"\n')
        ss_value = kwargs['ss_value']
    else:
        ss_value = ss

    def contract_2e(h1_h2, civec, norb, nelec, hdiag=None, **kwargs):
        if isinstance(nelec, (int, numpy.number)):
            sz = (nelec % 2) * .5
        else:
            sz = abs(nelec[0]-nelec[1]) * .5
        if ss_value is None:
            ss = sz*(sz+1)
        else:
            ss = ss_value

        h1, eri = h1_h2
        strs = civec._strs
        ndet = len(strs)
        if hdiag is None:
            hdiag = make_hdiag(h1, eri, strs, norb, nelec)
        ci1 = numpy.zeros_like(civec)
        ci2 = numpy.zeros_like(civec)

        h1 = numpy.asarray(h1, order='C')
        eri = numpy.asarray(eri, order='C')
        strs = numpy.asarray(strs, order='C')
        civec = numpy.asarray(civec, order='C')
        hdiag = numpy.asarray(hdiag, order='C')
        ci1 = numpy.asarray(ci1, order='C')
        ci2 = numpy.asarray(ci2, order='C')

        libhci.contract_h_c_ss_c(h1.ctypes.data_as(ctypes.c_void_p), 
                                 eri.ctypes.data_as(ctypes.c_void_p), 
                                 ctypes.c_int(norb), 
                                 ctypes.c_int(nelec[0]), 
                                 ctypes.c_int(nelec[1]), 
                                 strs.ctypes.data_as(ctypes.c_void_p), 
                                 civec.ctypes.data_as(ctypes.c_void_p), 
                                 hdiag.ctypes.data_as(ctypes.c_void_p), 
                                 ctypes.c_ulonglong(ndet), 
                                 ci1.ctypes.data_as(ctypes.c_void_p),
                                 ci2.ctypes.data_as(ctypes.c_void_p))

        if ss < sz*(sz+1)+.1:
# (S^2-ss)|Psi> to shift state other than the lowest state
            ci2 -= ss * civec
        else:
# (S^2-ss)^2|Psi> to shift states except the given spin.
# It still relies on the quality of initial guess
            tmp = ci2.copy()
            tmp -= ss * civec
            ci2 = -ss * tmp
            ci2 += myci.contract_ss(as_SCIvector_if_not(tmp, strs), norb, nelec)
            tmp = None
        ci2 *= shift
        ci1 += ci2

        return cistring.as_SCIvector_if_not(ci1, strs)

    myci.contract_2e = contract_2e
    return myci

class CI(direct_spin1.FCISolver):
    def __init__(self, mol=None):
        direct_spin1.FCISolver.__init__(self, mol)
        self.conv_tol = 1e-6
        self.model = 'fci'
        self.noise = False
##################################################
# don't modify the following attributes, they are not input options
        self._ndets = None
        self._strs = None
        self._keys = set(self.__dict__.keys())

    def dump_flags(self, verbose=None):
        if self.verbose >= logger.WARN:
            self.check_sanity()
        direct_spin1.FCISolver.dump_flags(self, verbose)
        #logger.info(self, 'Number of electrons = %s', self.nelec)
        #logger.info(self, 'Number of orbitals = %3d', self.norb)
        logger.info(self, 'CI molde to be solved of orbitals = %s', self.model)
        logger.info(self, 'Add noise to CI vector = %s', self.noise)
        return self

    # define absorb_h1e for compatibility to other FCI solver
    def absorb_h1e(h1, eri, *args, **kwargs):
        return (h1, eri)

    def contract_2e(self, h1_h2, civec, norb, nelec, hdiag=None, **kwargs):
        if hasattr(civec, '_strs'):
            self._strs = civec._strs
        else:
            assert(civec.size == len(self._strs))
            civec = as_SCIvector(civec, self._strs)
        return contract_2e(h1_h2, civec, norb, nelec, hdiag, **kwargs)

    def contract_ss(self, civec, norb, nelec):
        if hasattr(civec, '_strs'):
            self._strs = civec._strs
        else:
            assert(civec.size == len(self._strs))
            civec = as_SCIvector(civec, self._strs)
        return contract_ss(civec, norb, nelec)

    def spin_square(self, civec, norb, nelec):
        if hasattr(civec, '_strs'):
            self._strs = civec._strs
        else:
            assert(civec.size == len(self._strs))
            civec = as_SCIvector(civec, self._strs)
        return spin_square(civec, norb, nelec)

    def make_hdiag(self, h1e, eri, strs, norb, nelec):
        return make_hdiag(h1e, eri, strs, norb, nelec)

    def make_rdm12s(self, civec, norb, nelec):

        if hasattr(civec, '_strs'):
            self._strs = civec._strs
        else:
            assert(civec.size == len(self._strs))
            civec = cistring.as_SCIvector(civec, self._strs)

        return rdm.make_rdm12s(civec, norb, nelec)

    def make_rdm12(self, civec, norb, nelec):

        if hasattr(civec, '_strs'):
            self._strs = civec._strs
        else:
            assert(civec.size == len(self._strs))
            civec = cistring.as_SCIvector(civec, self._strs)

        return rdm.make_rdm12(civec, norb, nelec)

    kernel = kernel

CI = CI

if __name__ == '__main__':
    from pyscf import gto, scf, fci, mcscf
    mol = gto.Mole()
    mol.basis = 'sto-6g'
    mol.atom = '''
N 0.0000  0.0000  0.0000
N 0.0000  0.0000  1.1000
    '''
#Li 0.0000  0.0000  0.0000
#Li 0.0000  0.0000  2.6500
    mol.verbose = 4
    mol.spin = 0
    mol.charge = 0
    mol.symmetry = 0
    mol.build()

    mf = scf.RHF(mol)
    mf.kernel()

    #ncore = 0
    #e_core = mol.energy_nuc()
    #core_idx = numpy.arange(ncore)
    #ncore = core_idx.size
    #cas_idx = numpy.arange(ncore, numpy.shape(mf.mo_coeff)[1])
    #hcore = mf.get_hcore()
    #core_dm = numpy.dot(mf.mo_coeff[:, core_idx], mf.mo_coeff[:, core_idx].T)*2.0
    #e_core += numpy.einsum('ij,ji', core_dm, hcore)
    #corevhf = scf.hf.get_veff(mol, core_dm)
    #e_core += numpy.einsum('ij,ji', core_dm, corevhf)*0.5
    #h1e = reduce(numpy.dot, (mf.mo_coeff[:, cas_idx].T, hcore + corevhf, mf.mo_coeff[:, cas_idx]))
    #h2e = ao2mo.full(mf._eri, mf.mo_coeff[:, cas_idx])
    #nelec = mol.nelectron - ncore*2
    #nelec = (3,3)
    #norb = cas_idx.size

    roots = 2
    ncore = 2
    norb = mf.mo_coeff.shape[0] - ncore
    nelec = mol.nelectron - ncore*2
    nelec = (nelec/2,nelec/2)
    mc = mcscf.CASCI(mf, norb, nelec)
    h1e, e_core = mc.get_h1eff()
    h2e = mc.get_h2eff()

    mc = CI()
    mc = fix_spin(mc, ss=0, shif=0.8)
    mc.nroots = roots
    mc.verbose = 4
    e, civec = mc.kernel(h1e, h2e, norb, nelec, ecore=e_core, verbose=5)
    logger.info(mc,"* CI Energy : %s" % e)    
    for i in range(roots):
        norm = numpy.einsum('i,i->', civec[i][:], civec[i][:])
        s = mc.spin_square(civec[i], norb, nelec)
        logger.info(mc,"* Norm info for state %d : %s" % (i,norm))    
        logger.info(mc,"* Spin info for state %d : %s" % (i,s))    
    rdm1, rdm2 = mc.make_rdm12(civec[0], norb, nelec) 
    #print rdm1

    #t_start = time.time()
    #cisolver = fci.FCI(mol)
    #cisolver = fci.direct_spin1_symm.FCI(mol)
    #cisolver = fci.direct_spin1.FCI(mol)
    #cisolver.verbose = 4
    #cisolver.nroots = 8
    #e = cisolver.kernel(h1e, h2e, norb, nelec, ecore=e_core)[0]
    #t_current = time.time() - t_start
    #logger.info(cisolver,"* FCI Energy : %s" % e)    
    #logger.info(cisolver,"* Timing for solving the eigenvalue problem: %10.3f", t_current)

    t_start = time.time()
    mycas = mcscf.CASCI(mf, norb, nelec)
    mycas.fcisolver.nroots = roots
    mycas.verbose = 0
    mycas.fix_spin_(ss=0,shift=0.2)
    mycas.kernel()[0]
    rdm1, rdm2 = mc.make_rdm12(mycas.ci[0], norb, nelec) 
    #print rdm1
    t_current = time.time() - t_start
    logger.info(mc,"* FCI Energy : %s" % e)    
    logger.info(mc,"* Timing for solving the eigenvalue problem: %10.3f", t_current)

