#!/usr/bin/env python

import numpy, scipy

def imom_occ_(mf, occorb, setocc, mo_coeff):
    '''Use i-maximum overlap method to determine occupation number for each orbital in every
    iteration. It can be applied to unrestricted HF/KS and restricted open-shell
    HF/KS.'''
    from pyscf.scf import uhf, rohf
    if isinstance(mf, uhf.UHF):
        coef_occ_a = occorb[0][:, setocc[0]>0]
        coef_occ_b = occorb[1][:, setocc[1]>0]
    elif isinstance(mf, rohf.ROHF):
        if mf.mol.spin != int(numpy.sum(setocc[0]) - numpy.sum(setocc[1])) :
            raise ValueError('Wrong occupation setting for restricted open-shell calculation.') 
        coef_occ_a = occorb[:, setocc[0]>0]
        coef_occ_b = occorb[:, setocc[1]>0]
    else:
        raise AssertionError('Can not support this class of instance.')
    log = logger.Logger(mf.stdout, mf.verbose)
    def get_occ(mo_energy=None, mo_coeff=mo_coeff):
        if mo_energy is None: mo_energy = mf.mo_energy
        if isinstance(mf, rohf.ROHF): mo_coeff = numpy.array([mo_coeff, mo_coeff])
        mo_occ = numpy.zeros_like(setocc)
        nocc_a = int(numpy.sum(setocc[0]))
        nocc_b = int(numpy.sum(setocc[1]))
        s_a = reduce(numpy.dot, (coef_occ_a.T, mf.get_ovlp(), mo_coeff[0])) 
        s_b = reduce(numpy.dot, (coef_occ_b.T, mf.get_ovlp(), mo_coeff[1]))
        #choose a subset of mo_coeff, which maximizes <old|now>
        idx_a = numpy.argsort(numpy.einsum('ij,ij->j', s_a, s_a))
        idx_b = numpy.argsort(numpy.einsum('ij,ij->j', s_b, s_b))
        mo_occ[0][idx_a[-nocc_a:]] = 1.
        mo_occ[1][idx_b[-nocc_b:]] = 1.

        if mf.verbose >= logger.DEBUG:
            log.info(' New alpha occ pattern: %s', mo_occ[0])
            log.info(' New beta occ pattern: %s', mo_occ[1])
        if mf.verbose >= logger.DEBUG1:
            if mo_energy.ndim == 2:
                log.info(' Current alpha mo_energy(sorted) = %s', mo_energy[0])
                log.info(' Current beta mo_energy(sorted) = %s', mo_energy[1])
            elif mo_energy.ndim == 1:
                log.info(' Current mo_energy(sorted) = %s', mo_energy)

        if (int(numpy.sum(mo_occ[0])) != nocc_a):
            log.error('mom alpha electron occupation numbers do not match: %d, %d',
                      nocc_a, int(numpy.sum(mo_occ[0])))
        if (int(numpy.sum(mo_occ[1])) != nocc_b):
            log.error('mom alpha electron occupation numbers do not match: %d, %d',
                      nocc_b, int(numpy.sum(mo_occ[1])))

        #output 1-dimension occupation number for restricted open-shell
        if isinstance(mf, rohf.ROHF): mo_occ = mo_occ[0, :] + mo_occ[1, :]
        return mo_occ
    mf.get_occ = get_occ
    return mf
imom_occ = imom_occ_

