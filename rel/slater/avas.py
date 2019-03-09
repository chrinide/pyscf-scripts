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
#         Elvira R. Sayfutyarova
#

'''
Automated construction of molecular active spaces from atomic valence orbitals.
Ref. arXiv:1701.07862 [physics.chem-ph]
'''

import re
from functools import reduce
import numpy
import scipy.linalg
from pyscf import lib
from pyscf import gto
from pyscf import scf
from pyscf.lib import logger
from pyscf import __config__

THRESHOLD_OCC = getattr(__config__, 'mcscf_avas_threshold_occ', 0.1)
THRESHOLD_VIR = getattr(__config__, 'mcscf_avas_threshold_vir', 0.01)
MINAO = getattr(__config__, 'mcscf_avas_minao', 'ano')
CANONICALIZE = getattr(__config__, 'mcscf_avas_canonicalize', True)
NCORE = getattr(__config__, 'mcscf_avas_ncore', 0)


def kernel(mf, aolabels, threshold_occ=THRESHOLD_OCC, threshold_vir=THRESHOLD_VIR, 
           minao=MINAO, canonicalize=CANONICALIZE, ncore=NCORE, verbose=None):
    '''AVAS method to construct mcscf active space.
    Ref. arXiv:1701.07862 [physics.chem-ph]

    Args:
        mf : an :class:`SCF` object

        aolabels : string or a list of strings
            AO labels for AO active space

    Kwargs:
        threshold : float
            Tructing threshold of the AO-projector above which AOs are kept in
            the active space.
        minao : str
            A reference AOs for AVAS.
        canonicalize : bool
            Orbitals defined in AVAS method are local orbitals.  Symmetrizing
            the core, active and virtual space.
        ncore : integer
            Number of core orbitals to exclude from the AVAS method.

    Returns:
        active-space-size, #-active-electrons, orbital-initial-guess-for-CASCI/CASSCF

    Examples:

    >>> from pyscf import gto, scf, mcscf
    >>> from pyscf.mcscf import avas
    >>> mol = gto.M(atom='Cr 0 0 0; Cr 0 0 1.6', basis='ccpvtz')
    >>> mf = scf.RHF(mol).run()
    >>> ncas, nelecas, mo = avas.avas(mf, ['Cr 3d', 'Cr 4s'])
    >>> mc = mcscf.CASSCF(mf, ncas, nelecas).run(mo)
    '''
    if isinstance(verbose, logger.Logger):
        log = verbose
    elif verbose is not None:
        log = logger.Logger(mf.stdout, verbose)
    else:
        log = logger.Logger(mf.stdout, mf.verbose)

    mo_coeff = mf.mo_coeff
    mo_occ = mf.mo_occ
    mo_energy = mf.mo_energy
    nocc = numpy.count_nonzero(mo_occ != 0)

    log.info('\n** AVAS **')
    log.info('  Total number of HF MOs  is equal to    %d' ,mo_coeff.shape[1])
    log.info('  Number of occupied HF MOs is equal to  %d', nocc)
    log.info('  Number of core HF MOs is equal to  %d', ncore)

    mol = mf.mol
    pmol = mol.copy()
    pmol.atom = mol._atom
    pmol.unit = 'B'
    pmol.symmetry = False
    pmol.basis = minao
    pmol.build(False, False)

    baslst = pmol.rsearch_ao_label(aolabels)
    log.info('Reference AO indices for %s %s: %s', minao, aolabels, baslst)

    s22 = pmol.intor_symmetric('int1e_ovlp_spinor')[baslst][:,baslst]
    s21 = gto.mole.intor_cross('int1e_ovlp_spinor', pmol, mol)[baslst]
    s21 = numpy.dot(s21, mo_coeff[:, ncore:])
    sa = s21.conj().T.dot(lib.cho_solve(s22, s21))

    wocc, u = numpy.linalg.eigh(sa[:(nocc-ncore), :(nocc-ncore)])
    log.info('Option 2: threshold_occ %s', threshold_occ)
    log.info('Option 2: threshold_vir %s', threshold_vir)
    ncas_occ = (wocc > threshold_occ).sum()
    nelecas = (mol.nelectron - ncore) - (wocc < threshold_occ).sum()
    if ncore > 0: mofreeze = mo_coeff[:,:ncore]
    mocore = mo_coeff[:,ncore:nocc].dot(u[:,wocc<threshold_occ])
    mocas = mo_coeff[:,ncore:nocc].dot(u[:,wocc>threshold_occ])

    wvir, u = numpy.linalg.eigh(sa[(nocc-ncore):,(nocc-ncore):])
    ncas_vir = (wvir > threshold_vir).sum()
    mocas = numpy.hstack((mocas, mo_coeff[:,nocc:].dot(u[:,wvir>threshold_vir])))
    movir = mo_coeff[:,nocc:].dot(u[:,wvir<threshold_vir])
    ncas = mocas.shape[1]

    log.debug('projected occ eig %s', wocc[::-1])
    log.debug('projected vir eig %s', wvir[::-1])
    log.info('Active from occupied = %d , eig %s', ncas_occ, wocc[wocc>threshold_occ][::-1])
    log.info('Inactive from occupied = %d', mocore.shape[1])
    log.info('Active from unoccupied = %d , eig %s', ncas_vir, wvir[wvir>threshold_vir][::-1])
    log.info('Inactive from unoccupied = %d', movir.shape[1])
    log.info('Dimensions of active %d', ncas)
    log.info('Number of electrons %d', nelecas)

    if canonicalize:
        ovlp = mol.intor_symmetric('int1e_ovlp_spinor')
        def trans(c):
            if c.shape[1] == 0:
                return c
            else:
                csc = reduce(numpy.dot, (c.T, ovlp, mo_coeff))
                fock = numpy.dot(csc*mo_energy, csc.T)
                e, u = scipy.linalg.eigh(fock)
                return numpy.dot(c, u)
        if ncore > 0:
           mo = numpy.hstack([trans(mofreeze), trans(mocore), trans(mocas), trans(movir)])
        else:
           mo = numpy.hstack([trans(mocore), trans(mocas), trans(movir)])
    else:
        mo = numpy.hstack((mocore, mocas, movir))
    return ncas, nelecas, mo
avas = kernel

del(THRESHOLD_OCC, THRESHOLD_VIR, MINAO, CANONICALIZE)


if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf
    from pyscf import x2c

    mol = gto.M(
    verbose = 4,
    atom = '''
           H    0.000000,  0.500000,  1.5   
           Pb   0.000000,  0.000000,  1.
           O    0.000000,  0.000000, -1.
           H    0.000000, -0.500000, -1.5''',
        basis = 'dzp-dk',
    )

    mf = x2c.UHF(mol)
    mf.scf()

    aolst = ['O 2p', 'Pb 6p']
    ncas, nelecas, mo = avas(mf, aolst, verbose=4, ncore=80)

