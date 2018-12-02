#!/usr/bin/env python
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
from pyscf import lib
from pyscf import ao2mo
from pyscf.lib import logger
from pyscf.fci import direct_spin1

libhci = lib.load_library('libhci')

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
    diagj = numpy.einsum('iijj->ij',eri)
    diagk = numpy.einsum('ijji->ij',eri)

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
           nroots=None, davidson_only=None, max_iter=None,
           max_memory=None, verbose=None, ecore=0, **kwargs):

    if tol is None: tol = myci.conv_tol
    if lindep is None: lindep = myci.lindep
    if max_cycle is None: max_cycle = myci.max_cycle
    if max_space is None: max_space = myci.max_space
    if max_memory is None: max_memory = myci.max_memory
    if nroots is None: nroots = myci.nroots
    if max_iter is None: max_iter = myci.max_iter
    if verbose is None: verbose = logger.Logger(myci.stdout, myci.verbose)
    tol_residual = getattr(fci, 'conv_tol_residual', None)
    myci.dump_flags()

    nelec = direct_spin1._unpack_nelec(nelec, myci.spin)
    eri = ao2mo.restore(1, eri, norb)
    eri = eri.ravel()

    # Initial guess
    if ci0 is None:
        dets = cistring.gen_full_space(range(norb), nelec) 
        ndets = dets.shape[0]
        ci0 = [cistring.as_SCIvector(numpy.zeros(ndets), dets)]
        ci0[0][0] = 1.0
    else:
        assert(nroots == len(ci0))

    def hop(c):
        hc = myci.contract_2e((h1e, eri), cistring.as_SCIvector(c, ci_strs), norb, nelec, hdiag)
        return hc.ravel()
    precond = lambda x, e, *args: x/(hdiag-e+myci.level_shift)

    ci_strs = ci0[0]._strs
    logger.info(myci, 'CI in the selected space')
    logger.info(myci, 'Number of CI configurations: %d', ci_strs.shape[0])
    t_start = time.time()
    hdiag = myci.make_hdiag(h1e, eri, ci_strs, norb, nelec)
    with lib.with_omp_threads(myci.threads):
        #e, c = myci.eig(hop, ci0, precond, tol=tol, lindep=lindep,
        #                max_cycle=max_cycle, max_space=max_space, nroots=nroots,
        #                max_memory=max_memory, verbose=verbose, **kwargs)
        #
        e, c = myci.eig(hop, ci0, precond, tol=tol, lindep=lindep,
                       max_cycle=max_cycle, max_space=max_space, nroots=nroots,
                       max_memory=max_memory, verbose=verbose, follow_state=True,
                       tol_residual=tol_residual, **kwargs)
    #TODO:chech myci.conv
    t_current = time.time() - t_start
    logger.info(myci, 'Timing for solving the eigenvalue problem: %10.3f', t_current)
    if not isinstance(c, (tuple, list)):
        c = [c]
        e = [e]
    logger.info(myci, 'CI  E = %s', numpy.array(e)+ecore)

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

        return as_SCIvector_if_not(ci1, strs)

    myci.contract_2e = contract_2e
    return myci

class CI(direct_spin1.FCISolver):
    def __init__(self, mol=None):
        direct_spin1.FCISolver.__init__(self, mol)
        self.conv_tol = 1e-6
        self.nroots = 1
        self.max_iter = 4
        self.max_memory = 1000

##################################################
# don't modify the following attributes, they are not input options
        self.ndets = None
        self._strs = None
        self._keys = set(self.__dict__.keys())

    def dump_flags(self, verbose=None):
        if self.verbose >= logger.WARN:
            self.check_sanity()
        direct_spin1.FCISolver.dump_flags(self, verbose)
        #logger.info(self, 'Number of electrons = %s', nelec)
        #logger.info(self, 'Number of orbitals = %3d', norb)
        #logger.info(self, 'Number of roots = %3d', nroots)
        #logger.info(self, 'Number of dets = %3d', nroots)
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
            civec = as_SCIvector(civec, self._strs)

        return make_rdm12s(civec, norb, nelec)

    kernel = kernel

CI = CI

if __name__ == '__main__':
    from pyscf import gto, scf, fci
    mol = gto.Mole()
    mol.basis = 'aug-cc-pvtz'
    mol.atom = '''
    H  0.0000  0.0000  0.0000
    H  0.0000  0.0000  3.7500
    '''
    mol.verbose = 4
    mol.spin = 0
    mol.charge = 0
    mol.symmetry = 1
    mol.build()

    mf = scf.RHF(mol)
    mf.kernel()

    nelec = (1,1)
    nao, nmo = mf.mo_coeff.shape
    eri = ao2mo.kernel(mf._eri, mf.mo_coeff[:,:nmo], compact=False)
    eri = eri.reshape(nmo,nmo,nmo,nmo)
    h1 = reduce(numpy.dot, (mf.mo_coeff[:,:nmo].T, mf.get_hcore(), mf.mo_coeff[:,:nmo]))

    mc = CI()
    mc.nroots = 2
    mc.verbose = 4
    e = mc.kernel(h1, eri, nmo, nelec, verbose=5)[0]
    e += mf.energy_nuc()
    print('E(CI) = %s' % e)

    cisolver = fci.FCI(mol, mf.mo_coeff)
    cisolver.nroots = 2
    print('E(FCI) = %s' % cisolver.kernel()[0])
