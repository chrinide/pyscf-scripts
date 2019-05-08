#!/usr/bin/env python
# $Id$
# -*- coding: utf-8
# Author: Qiming Sum

import time
import tempfile
from functools import reduce
import numpy
import h5py
from pyscf import lib
from pyscf.lib import logger
import pyscf.ao2mo
from pyscf.ao2mo import r_outcore

# JCP, 139, 014108

def ao2mo(mf, mo_coeff, erifile):
    n4c, nmo = mo_coeff.shape
    n2c = n4c // 2
    nN = nmo // 2
    nocc = int(mf.mo_occ.sum())
    nP = nN + nocc
    nvir = nmo - nP
    c = lib.param.LIGHT_SPEED
    ol = mo_coeff[:n2c]
    os = mo_coeff[n2c:] * (.5/c)
    ol0 = ol[:,nN:nP]
    os0 = os[:,nN:nP]
    ol1 = ol[:,:nP]
    os1 = os[:,:nP]
    ol2 = ol[:,nP:]
    os2 = os[:,nP:]
    ol3 = ol[:,:nN]
    os3 = os[:,:nN]
    ol4 = ol[:,nN:]
    os4 = os[:,nN:]

    def run(mos, intor, dst, bound, blksize):
        logger.debug(mf, 'blksize = %d', blksize)
        r_outcore.general(mf.mol, mos, erifile,
                          dataname='tmp', intor=intor,
                          verbose=mf.verbose)
        with h5py.File(erifile) as feri:
            for i0, i1 in prange(0, bound, blksize):
                logger.debug(mf, 'load %s %d', intor, i0)
                buf = feri[dst][i0:i1]
                buf += feri['tmp'][i0:i1]
                feri[dst][i0:i1] = buf
                buf = None

    r_outcore.general(mf.mol, (ol0,ol2,ol1,ol2), erifile,
                      dataname='ij', intor='cint2e',
                      verbose=mf.verbose)
    blksize = max(1, int(512e6/16/(nvir*nP*nvir))) * nvir
    run((os0,os2,os1,os2), 'cint2e_spsp1spsp2', 'ij', nocc*nvir, blksize)
    run((os0,os2,ol1,ol2), 'cint2e_spsp1'     , 'ij', nocc*nvir, blksize) #ssll
    run((ol0,ol2,os1,os2), 'cint2e_spsp2'     , 'ij', nocc*nvir, blksize)

    r_outcore.general(mf.mol, (ol0,ol3,ol4,ol3), erifile,
                      dataname='ia', intor='cint2e',
                      verbose=mf.verbose)
    blksize = max(1, int(512e6/16/(nN*nN*nN))) * nN
    run((os0,os3,os4,os3), 'cint2e_spsp1spsp2', 'ia', nocc*nN, blksize)
    run((os0,os3,ol4,ol3), 'cint2e_spsp1'     , 'ia', nocc*nN, blksize)
    run((ol0,ol3,os4,os3), 'cint2e_spsp2'     , 'ia', nocc*nN, blksize)


def ao2mo_gaunt(mf, mo_coeff, erifile):
    ao2mo(mf, mo_coeff, erifile)

    n4c, nmo = mo_coeff.shape
    n2c = n4c // 2
    nN = nmo // 2
    nocc = int(mf.mo_occ.sum())
    nP = nN + nocc
    nvir = nmo - nP
    c = lib.param.LIGHT_SPEED
    ol = mo_coeff[:n2c]
    os = mo_coeff[n2c:] * (.5/c)
    ol0 = ol[:,nN:nP]
    os0 = os[:,nN:nP]
    ol1 = ol[:,:nP]
    os1 = os[:,:nP]
    ol2 = ol[:,nP:]
    os2 = os[:,nP:]
    ol3 = ol[:,:nN]
    os3 = os[:,:nN]
    ol4 = ol[:,nN:]
    os4 = os[:,nN:]

    def run(mos, intor, dst, bound, blksize):
        logger.debug(mf, 'blksize = %d', blksize)
        r_outcore.general(mf.mol, mos, erifile,
                          dataname='tmp', intor=intor,
                          aosym='s1', verbose=mf.verbose)
        with h5py.File(erifile) as feri:
            for i0, i1 in prange(0, bound, blksize):
                logger.debug(mf, 'load %s %d', intor, i0)
                buf = feri[dst][i0:i1]
                buf -= feri['tmp'][i0:i1]
                feri[dst][i0:i1] = buf
                buf = None

    blksize = max(1, int(512e6/16/(nvir*nP*nvir))) * nvir
    run((ol0,os2,ol1,os2), 'cint2e_ssp1ssp2', 'ij', nocc*nvir, blksize)
    run((ol0,os2,os1,ol2), 'cint2e_ssp1sps2', 'ij', nocc*nvir, blksize)
    run((os0,ol2,ol1,os2), 'cint2e_sps1ssp2', 'ij', nocc*nvir, blksize)
    run((os0,ol2,os1,ol2), 'cint2e_sps1sps2', 'ij', nocc*nvir, blksize)

    blksize = max(1, int(512e6/16/(nN*nN*nN))) * nN
    run((ol0,os3,ol4,os3), 'cint2e_ssp1ssp2', 'ia', nocc*nN, blksize)
    run((ol0,os3,os4,ol3), 'cint2e_ssp1sps2', 'ia', nocc*nN, blksize)
    run((os0,ol3,ol4,os3), 'cint2e_sps1ssp2', 'ia', nocc*nN, blksize)
    run((os0,ol3,os4,ol3), 'cint2e_sps1sps2', 'ia', nocc*nN, blksize)

def ao2mo_breit(mf, mo_coeff, erifile):
    ao2mo(mf, mo_coeff, erifile)

    n4c, nmo = mo_coeff.shape
    n2c = n4c // 2
    nN = nmo // 2
    nocc = int(mf.mo_occ.sum())
    nP = nN + nocc
    nvir = nmo - nP
    c = lib.param.LIGHT_SPEED
    ol = mo_coeff[:n2c]
    os = mo_coeff[n2c:] * (.5/c)
    ol0 = ol[:,nN:nP]
    os0 = os[:,nN:nP]
    ol1 = ol[:,:nP]
    os1 = os[:,:nP]
    ol2 = ol[:,nP:]
    os2 = os[:,nP:]
    ol3 = ol[:,:nN]
    os3 = os[:,:nN]
    ol4 = ol[:,nN:]
    os4 = os[:,nN:]

    def run(mos, intor, dst, bound, blksize):
        logger.debug(mf, 'blksize = %d', blksize)
        r_outcore.general(mf.mol, mos, erifile,
                          dataname='tmp', intor=intor,
                          aosym='s1', verbose=mf.verbose)
        with h5py.File(erifile) as feri:
            for i0, i1 in prange(0, bound, blksize):
                logger.debug(mf, 'load %s %d', intor, i0)
                buf = feri[dst][i0:i1]
                buf -= feri['tmp'][i0:i1]
                feri[dst][i0:i1] = buf
                buf = None

    blksize = max(1, int(512e6/16/(nvir*nP*nvir))) * nvir
    run((ol0,os2,ol1,os2), 'cint2e_breit_ssp1ssp2', 'ij', nocc*nvir, blksize)
    run((ol0,os2,os1,ol2), 'cint2e_breit_ssp1sps2', 'ij', nocc*nvir, blksize)
    run((os0,ol2,ol1,os2), 'cint2e_breit_sps1ssp2', 'ij', nocc*nvir, blksize)
    run((os0,ol2,os1,ol2), 'cint2e_breit_sps1sps2', 'ij', nocc*nvir, blksize)

    blksize = max(1, int(512e6/16/(nN*nN*nN))) * nN
    run((ol0,os3,ol4,os3), 'cint2e_breit_ssp1ssp2', 'ia', nocc*nN, blksize)
    run((ol0,os3,os4,ol3), 'cint2e_breit_ssp1sps2', 'ia', nocc*nN, blksize)
    run((os0,ol3,ol4,os3), 'cint2e_breit_sps1ssp2', 'ia', nocc*nN, blksize)
    run((os0,ol3,os4,ol3), 'cint2e_breit_sps1sps2', 'ia', nocc*nN, blksize)

def make_eq146(mf, erifile):
    mo_energy = mf.mo_energy
    nocc = int(mf.mo_occ.sum())
    nmo = mo_energy.size
    nN = nmo // 2
    nP = nN + nocc
    nvir = nmo - nP

    dm = mf.make_rdm1()
    U = reduce(numpy.dot, (mf.mo_coeff.T.conj(), mf.get_veff(mf.mol, dm),
                           mf.mo_coeff))

    e2a = e2b = 0
    eia = lib.direct_sum('i-a->ia', mo_energy[:nN], mo_energy[nP:]) # eq 67
    e2a =-numpy.einsum('ia,ai,ia->', U[:nN,nP:], U[nP:,:nN], 1/eia)
    eij = lib.direct_sum('i-j->ij', mo_energy[:nN], mo_energy[nN:nP])
    e2b-= numpy.einsum('ia,ai,ia->', U[:nN,nN:nP], U[nN:nP,:nN], 1/eij)
    logger.info(mf, 'Eq146 = %s + %s', e2a, e2b)
    e2 = e2a + e2b
    return e2

def make_eq147(mf, erifile):
    mo_energy = mf.mo_energy
    mo_coeff = mf.mo_coeff
    nocc = int(mf.mo_occ.sum())
    nmo = mo_energy.size
    nN = nmo // 2
    nP = nN + nocc
    nvir = nmo - nP

    dm =(numpy.dot(mo_coeff[:,nN:], mo_coeff[:,nN:].T.conj())
       - numpy.dot(mo_coeff[:,:nN], mo_coeff[:,:nN].T.conj()))
    Q = reduce(numpy.dot, (mo_coeff.T.conj(), mf.get_veff(mf.mol, dm),
                           mo_coeff))
    Q *= -.5

    dm = mf.make_rdm1()
    U = vhf = reduce(numpy.dot, (mo_coeff.T.conj(), mf.get_veff(mf.mol, dm),
                                 mo_coeff))

    eia = 1./lib.direct_sum('i-a->ia', mo_energy[nN:nP], mo_energy[nP:]) # eq 67
    e2 = numpy.einsum('ia,ai,ia->', Q[nN:nP,nP:], Q[nP:,nN:nP], eia)

    eia = 1./lib.direct_sum('i-a->ia', mo_energy[:nN], mo_energy[nP:])
    e2+= numpy.einsum('ia,ai,ia->', vhf[:nN,nP:], Q[nP:,:nN], eia)
    e2+= numpy.einsum('ia,ai,ia->', Q[:nN,nP:], vhf[nP:,:nN], eia)

    eij = 1./lib.direct_sum('i-j->ij', mo_energy[nN:nP], mo_energy[:nN])
    e2-= numpy.einsum('ji,ij,ij->', Q[:nN,nN:nP], Q[nN:nP,:nN], eij)
    e2+= numpy.einsum('ji,ij,ij->', U[:nN,nN:nP], Q[nN:nP,:nN], eij)
    e2+= numpy.einsum('ji,ij,ij->', Q[:nN,nN:nP], U[nN:nP,:nN], eij)
    return e2

def make_eq64(mf, erifile):
    mo_energy = mf.mo_energy
    mo_coeff = mf.mo_coeff
    nocc = int(mf.mo_occ.sum())
    nmo = mo_energy.size
    nN = nmo // 2
    nP = nN + nocc
    nvir = nmo - nP

    dm =(numpy.dot(mo_coeff[:,nN:], mo_coeff[:,nN:].T.conj())
       - numpy.dot(mo_coeff[:,:nN], mo_coeff[:,:nN].T.conj()))
    Q = reduce(numpy.dot, (mo_coeff.T.conj(), mf.get_veff(mf.mol, dm),
                           mo_coeff))
    Q *= -.5

    dm = mf.make_rdm1()
    U = vhf = reduce(numpy.dot, (mo_coeff.T.conj(), mf.get_veff(mf.mol, dm),
                                 mo_coeff))
    v1bar = Q - U
    eia = 1./lib.direct_sum('i-a->ia', mo_energy[nN:nP], mo_energy[nP:])
    e2 = numpy.einsum('ia,ai,ia->', Q[nN:nP,nP:], Q[nP:,nN:nP], eia)

    eia = 1./lib.direct_sum('i-a->ia', mo_energy[:nN], mo_energy[nP:])
    e2+= numpy.einsum('ia,ai,ia->', Q[:nN,nP:], Q[nP:,:nN], eia)
    e2-= numpy.einsum('ia,ai,ia->', v1bar[:nN,nP:], v1bar[nP:,:nN], eia)

    eij = 1./lib.direct_sum('i-j->ij', mo_energy[:nN], mo_energy[nN:nP])
    e2-= numpy.einsum('ij,ji,ij->', v1bar[:nN,nN:nP], v1bar[nN:nP,:nN], eij)
    return e2


def make_eq148(mf, erifile):
    mo_energy = mf.mo_energy
    nocc = int(mf.mo_occ.sum())
    nmo = mo_energy.size
    nN = nmo // 2
    nP = nN + nocc
    nvir = nmo - nP
    with h5py.File(erifile) as feri:
        emp2 = 0
        e2a = 0
        e2b = 0
        e2c = 0
        blksize = max(1, int(512e6/16/(nvir*nP*nvir)))
        for i0, i1 in prange(0, nocc, blksize):
            g = numpy.asarray(feri['ij'][i0*nvir:i1*nvir])
            g = g.reshape(-1,nvir,nP,nvir).transpose(0,2,1,3) # eq 6
            g = g - g.transpose(0,1,3,2) # eq 9

            g1 = g[:,nN:nP,:,:].copy()
            emp2 += .25 * numpy.einsum('ijab,ijab,ijab->', g1, g1.conj(),
                                       1./lib.direct_sum('i+j-a-b',
                                                        mo_energy[nN+i0:nN+i1],
                                                        mo_energy[nN:nP],
                                                        mo_energy[nP:  ],
                                                        mo_energy[nP:  ]))

            g1 = g[:,:nN,:,:]
            e2a += .5 * numpy.einsum('ijab,ijab,ijab->', g1, g1.conj(),
                                    1./lib.direct_sum('i+j-a-b',
                                                     mo_energy[nN+i0:nN+i1],
                                                     mo_energy[:nN  ],
                                                     mo_energy[nP:  ],
                                                     mo_energy[nP:  ]))


        blksize = max(1, int(512e6/16/(nN*nN*nN)))
        for i0, i1 in prange(0, nocc, blksize):
            g = numpy.asarray(feri['ia'][i0*nN:i1*nN])
            g = g.reshape(-1,nN,nN,nN).transpose(0,2,1,3) # eq 6
            g = g - g.transpose(0,1,3,2) # eq 9

            g1 = g[:,nocc:,:,:]
            e2b -= .5 * numpy.einsum('iamn,iamn,mnia->', g1, g1.conj(),
                                    1./lib.direct_sum('i+j-a-b',
                                                     mo_energy[:nN],
                                                     mo_energy[:nN],
                                                     mo_energy[nN+i0:nN+i1],
                                                     mo_energy[nP:]))

            g1 = g[:,:nocc,:,:]
            e2c -= .25 * numpy.einsum('ijmn,ijmn,mnij->', g1, g1.conj(),
                                     1./lib.direct_sum('i+j-a-b',
                                                      mo_energy[:nN],
                                                      mo_energy[:nN],
                                                      mo_energy[nN+i0:nN+i1],
                                                      mo_energy[nN:nP]))
    e2 = e2a + e2b + e2c
    e2 += emp2
    logger.info(mf, 'Eq148 = %s + %s + %s + %s', emp2, e2a, e2b, e2c)
    return emp2, e2

def prange(start, end, step):
    for i in range(start, end, step):
        yield i, min(i+step, end)

if __name__ == '__main__':
    from pyscf import scf, gto

    mol = gto.Mole()
    mol.basis = 'unc-dzp-dk'
    mol.atom = '''
    O      0.000000      0.000000      0.118351
    H      0.000000      0.761187     -0.469725
    H      0.000000     -0.761187     -0.469725
    '''
    mol.charge = 0
    mol.spin = 0
    mol.symmetry = 0
    mol.verbose = 4
    mol.build()

    mf = scf.DHF(mol)
    mf.with_ssss = True
    mf.kernel()

    erifile = lib.param.TMPDIR+'/qed2.h5'
    ao2mo(mf,mf.mo_coeff,erifile)

    e1 = make_eq64(mf,erifile)
    e2 = make_eq146(mf,erifile)
    e3 = make_eq147(mf,erifile)
    e4 = make_eq148(mf,erifile)
    print e4[0]

