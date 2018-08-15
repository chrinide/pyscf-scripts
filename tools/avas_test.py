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
MINAO = getattr(__config__, 'mcscf_avas_minao', 'minao')
WITH_IAO = getattr(__config__, 'mcscf_avas_with_iao', False)
OPENSHELL_OPTION = getattr(__config__, 'mcscf_avas_openshell_option', 2)
CANONICALIZE = getattr(__config__, 'mcscf_avas_canonicalize', True)
NCORE = getattr(__config__, 'mcscf_avas_ncore', 0)


def kernel(mf, locc, lvir, threshold_occ=THRESHOLD_OCC, threshold_vir=THRESHOLD_VIR, 
           minao=MINAO, with_iao=WITH_IAO, openshell_option=OPENSHELL_OPTION, 
           canonicalize=CANONICALIZE, ncore=NCORE, localize=None, verbose=None):

    from pyscf.tools import mo_mapping

    if isinstance(verbose, logger.Logger):
        log = verbose
    elif verbose is not None:
        log = logger.Logger(mf.stdout, verbose)
    else:
        log = logger.Logger(mf.stdout, mf.verbose)
    mol = mf.mol

    log.info('\n** AVAS **')
    if isinstance(mf, scf.uhf.UHF):
        raise NotImplementedError
    else:
        nao, nmo = mf.mo_coeff.shape
        nocc = mol.nelectron//2 - ncore
        nvir = nmo - nocc - ncore
        mo_coeff = mf.mo_coeff
        mo_occ = mf.mo_occ
        mo_energy = mf.mo_energy
        mo_coeff_core = mf.mo_coeff[:,:ncore]
        mo_coeff_occ = mf.mo_coeff[:,ncore:ncore+nocc]
        mo_coeff_vir = mf.mo_coeff[:,ncore+nocc:]
        mo_occ_core = mf.mo_occ[:ncore]
        if ncore > 0: mofreeze = mo_coeff_core
        mo_occ_occ = mf.mo_occ[ncore:ncore+nocc]
        mo_occ_vir = mf.mo_occ[ncore+nocc:]
        mo_energy_core = mf.mo_occ[:ncore]
        mo_enery_occ = mf.mo_occ[ncore:ncore+nocc]
        mo_energy_vir = mf.mo_occ[ncore+nocc:]
    ovlp = mol.intor_symmetric('int1e_ovlp')
    log.info('Total number of HF MOs is equal to %d' ,mf.mo_coeff.shape[1])
    log.info('Number of occupied HF MOs is equal to %d', nocc)
    log.info('Number of core HF MOs is equal to  %d', ncore)

    mol = mf.mol
    pmol = mol.copy()
    pmol.atom = mol._atom
    pmol.unit = 'B'
    pmol.symmetry = False
    pmol.basis = minao
    pmol.build(False, False)

    baslstocc = pmol.search_ao_label(locc)
    log.info('reference occ AO indices for %s %s: %s', minao, locc, baslstocc)
    baslstvir = pmol.search_ao_label(lvir)
    print lvir, pmol.search_ao_label(lvir) 
    log.info('reference vir AO indices for %s %s: %s', minao, lvir, baslstvir)

    s2occ = pmol.intor_symmetric('int1e_ovlp')[baslstocc][:,baslstocc]
    s21occ = gto.intor_cross('int1e_ovlp', pmol, mol)[baslstocc]
    s21occ = numpy.dot(s21occ, mo_coeff_occ)

    s2vir = pmol.intor_symmetric('int1e_ovlp')[baslstvir][:,baslstvir]
    s21vir = gto.intor_cross('int1e_ovlp', pmol, mol)[baslstvir]
    s21vir = numpy.dot(s21vir, mo_coeff_vir)

    saocc = s21occ.T.dot(scipy.linalg.solve(s2occ, s21occ, sym_pos=True))
    savir = s21vir.T.dot(scipy.linalg.solve(s2vir, s21vir, sym_pos=True))

    log.info('Threshold_occ %s', threshold_occ)
    wocc, u = numpy.linalg.eigh(saocc)
    log.info('projected occ eig %s', wocc[::-1])
    ncas_occ = (wocc > threshold_occ).sum()
    log.info('Active from occupied = %d , eig %s', ncas_occ, wocc[wocc>threshold_occ][::-1])
    nelecas = (mol.nelectron - ncore * 2) - (wocc < threshold_occ).sum() * 2
    mocore = mo_coeff_occ.dot(u[:,wocc<threshold_occ])
    log.info('Inactive from occupied = %d', mocore.shape[1])
    mocas = mo_coeff_occ.dot(u[:,wocc>threshold_occ])

    log.info('Threshold_vir %s', threshold_vir)
    wvir, u = numpy.linalg.eigh(savir)
    log.debug('projected vir eig %s', wvir[::-1])
    ncas_vir = (wvir > threshold_vir).sum()
    log.info('Active from unoccupied = %d , eig %s', ncas_vir, wvir[wvir>threshold_vir][::-1])
    mocas = numpy.hstack((mocas, mo_coeff_vir.dot(u[:,wvir>threshold_vir])))
    movir = mo_coeff_vir.dot(u[:,wvir<threshold_vir])
    log.info('Inactive from unoccupied = %d', movir.shape[1])
    ncas = mocas.shape[1]
    log.info('Dimensions of active %d', ncas)

    nalpha = (nelecas + mol.spin) // 2
    log.info('# of alpha electrons %d', nalpha)
    nbeta = nelecas - nalpha
    log.info('# of beta electrons %d', nbeta)

    if canonicalize:
        from pyscf.mcscf import dmet_cas
        def trans(c):
            if c.shape[1] == 0:
                return c
            else:
                csc = reduce(numpy.dot, (c.T, ovlp, mo_coeff))
                fock = numpy.dot(csc*mo_energy, csc.T)
                e, u = scipy.linalg.eigh(fock)
                return dmet_cas.symmetrize(mol, e, numpy.dot(c, u), ovlp, log)
        if ncore > 0:
           mo = numpy.hstack([trans(mofreeze), trans(mocore), trans(mocas), trans(movir)])
        else:
           mo = numpy.hstack([trans(mocore), trans(mocas), trans(movir)])
    else:
        if ncore > 0:
            mo = numpy.hstack((mofreeze, mocore, mocas, movir))
        else:
            mo = numpy.hstack((mocore, mocas, movir))

    return ncas, nelecas, mo
avas = kernel

del(THRESHOLD_OCC, THRESHOLD_VIR, MINAO, WITH_IAO, OPENSHELL_OPTION, CANONICALIZE)


if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf
    from pyscf import mcscf

    mol = gto.Mole()
    mol.basis = 'aug-cc-pvdz'
    mol.atom = '''
    Cr  0.0000  0.0000  0.0000
    Cr  0.0000  0.0000  1.6800
        '''
    mol.verbose = 4
    mol.spin = 0
    mol.charge = 0
    mol.symmetry = 1
    mol.build()

    mf = scf.RHF(mol)
    mf.kernel()

    ncore = 18

    aolst1 = ['Cr 3d']
    aolst2 = ['Cr 4s']
    aolst3 = ['Cr 4d']
    aolst4 = ['Cr 5s']
    aolstocc = aolst1 + aolst2
    aolstvir = aolst3 + aolst4
    ncas, nelecas, mo = kernel(mf, aolstocc, aolstvir, threshold_occ=0.1, \
                               threshold_vir=0.8, minao='ano', ncore=ncore)

    mc = mcscf.CASSCF(mf, ncas, nelecas)
    mc.max_cycle_macro = 250
    mc.max_cycle_micro = 7
    mc.fix_spin_(shift=.5, ss=0)
    mc.kernel(mo)

