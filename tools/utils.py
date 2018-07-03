#!/usr/bin/env python

import numpy, scipy
from pyscf.lib import logger

BETARAD = {
        'H'    :  0.180, 'He'   :  0.220, 'Li'   :  0.220, 'Be'   :  0.240,
        'B'    :  0.240, 'C'    :  0.260, 'N'    :  0.280, 'O'    :  0.300, 'F'    :  0.320,
        'Ne'   :  0.320, 'Na'   :  0.320, 'Mg'   :  0.320, 'Al'   :  0.320, 'Si'   :  0.320,
        'P'    :  0.320, 'S'    :  0.320, 'Cl'   :  0.320, 'Ar'   :  0.320, 'K'    :  0.320,
        'Ca'   :  0.320, 'Sc'   :  0.320, 'Ti'   :  0.320, 'V'    :  0.320, 'Cr'   :  0.320,
        'Mn'   :  0.320, 'Fe'   :  0.320, 'Co'   :  0.320, 'Ni'   :  0.320, 'Cu'   :  0.320,
        'Zn'   :  0.320, 'Ga'   :  0.320, 'Ge'   :  0.320, 'As'   :  0.320, 'Se'   :  0.320,
        'Br'   :  0.320, 'Kr'   :  0.320}

def smearing_(mf, sigma=None, method='fermi'):
    '''Fermi-Dirac or Gaussian smearing'''
    from pyscf.scf import uhf
    from pyscf.scf import hf
    mf_class = mf.__class__
    is_uhf = isinstance(mf, uhf.UHF)
    is_rhf = isinstance(mf, hf.RHF)
    mol_nelec = mf.mol.nelectron

    def fermi_smearing_occ(m, mo_energy, sigma):
        occ = numpy.zeros_like(mo_energy)
        de = (mo_energy - m) / sigma
        occ[de<40] = 1./(numpy.exp(de[de<40])+1.)
        return occ
    def gaussian_smearing_occ(m, mo_energy, sigma):
        return .5 - .5*scipy.special.erf((mo_energy-m)/sigma)

    def partition_occ(mo_occ, mo_energy):
        mo_occ = []
        p1 = 0
        for e in mo_energy:
            p0, p1 = p1, p1 + e.size
            occ = mo_occ[p0:p1]
            mo_occ.append(occ)
        return mo_occ

    def get_occ(mo_energy=None, mo_coeff=None):
        mo_occ = mf_class.get_occ(mf, mo_energy, mo_coeff)
        if mf.sigma == 0 or not mf.sigma or not mf.smearing_method:
            return mo_occ

        if is_uhf:
            nocc = mol_nelec
            mo_es = numpy.append(numpy.hstack(mo_energy[0]),
                                 numpy.hstack(mo_energy[1]))
        else:
            nocc = mol_nelec // 2
            mo_es = numpy.hstack(mo_energy)

        if mf.smearing_method.lower() == 'fermi':  # Fermi-Dirac smearing
            f_occ = fermi_smearing_occ
        else:  # Gaussian smearing
            f_occ = gaussian_smearing_occ

        mo_energy = numpy.sort(mo_es.ravel())
        fermi = mo_energy[nocc-1]
        sigma = mf.sigma
        def nelec_cost_fn(m):
            mo_occ = f_occ(m, mo_es, sigma)
            if not is_uhf:
                mo_occ *= 2
            return (mo_occ.sum() - mol_nelec)**2
        res = scipy.optimize.minimize(nelec_cost_fn, fermi, method='Powell')
        mu = res.x
        mo_occs = f = f_occ(mu, mo_es, sigma)

        if mf.smearing_method.lower() == 'fermi':
            f = f[(f>0) & (f<1)]
            mf.entropy = -(f*numpy.log(f) + (1-f)*numpy.log(1-f)).sum()
        else:
            mf.entropy = (numpy.exp(-((mo_es-mu)/mf.sigma)**2).sum()
                          / (2*numpy.sqrt(numpy.pi)))
        if not is_uhf:
            mo_occs *= 2
            mf.entropy *= 2

        if is_uhf:
            mo_occ = partition_occ(mo_occs, mo_energy)
        else:
            mo_occ = mo_occs

        logger.debug(mf, '    Fermi level %g  Sum mo_occ = %s  should equal nelec = %s',
                     fermi, mo_occs.sum(), mol_nelec)
        logger.info(mf, '    sigma = %g  Optimized mu = %.12g  entropy = %.12g',
                    mf.sigma, mu, mf.entropy)

        return mo_occ

    def get_grad_tril(mo_coeff, mo_occ, fock):
        f_mo = reduce(numpy.dot, (mo_coeff.T.conj(), fock, mo_coeff))
        nmo = f_mo.shape[0]
        return f_mo[numpy.tril_indices(nmo, -1)]

    def get_grad(mo_coeff, mo_occ, fock=None):
        if mf.sigma == 0 or not mf.sigma or not mf.smearing_method:
            return mf_class.get_grad(mf, mo_coeff, mo_occ, fock)
        if fock is None:
            dm1 = mf.make_rdm1(mo_coeff, mo_occ)
            fock = mf.get_hcore() + mf.get_veff(mf.cell, dm1)
        if is_uhf:
            ga = get_grad_tril(mo_coeff[0], mo_occ[0], fock[0])
            gb = get_grad_tril(mo_coeff[1], mo_occ[1], fock[1])
            return numpy.hstack((ga,gb))
        else:
            return get_grad_tril(mo_coeff, mo_occ, fock)

    def energy_tot(dm=None, h1e=None, vhf=None):
        e_tot = mf.energy_elec(dm, h1e, vhf)[0] + mf.energy_nuc()
        if (mf.sigma and mf.smearing_method and
            mf.entropy is not None and mf.verbose >= logger.INFO):
            mf.e_free = e_tot - mf.sigma * mf.entropy
            mf.e_zero = e_tot - mf.sigma * mf.entropy * .5
            logger.info(mf, '    Total E(T) = %.15g  Free energy = %.15g  E0 = %.15g',
                        e_tot, mf.e_free, mf.e_zero)
        return e_tot

    mf.sigma = sigma
    mf.smearing_method = method
    mf.entropy = None
    mf.e_free = None
    mf.e_zero = None
    mf._keys = mf._keys.union(['sigma', 'smearing_method',
                               'entropy', 'e_free', 'e_zero'])

    mf.get_occ = get_occ
    mf.energy_tot = energy_tot
    mf.get_grad = get_grad
    return mf

smearing = smearing_

