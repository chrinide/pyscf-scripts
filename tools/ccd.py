#!/usr/bin/env python

import numpy
from pyscf import ao2mo
from pyscf import lib
from pyscf.lib import logger
einsum = lib.einsum

def kernel(cc, eris, t1=None, t2=None, max_cycle=80, tol=1e-8, tolnormt=1e-8,
           verbose=logger.INFO):
    if verbose is None:
        verbose = cc.verbose
    if isinstance(verbose, logger.Logger):
        log = verbose
    else:
        log = logger.Logger(cc.stdout, verbose)

    t1, t2 = cc.get_init_guess(eris)[1:]
    eold = 0
    eccsd = 0
    if cc.diis:
        adiis = lib.diis.DIIS(cc, cc.diis_file)
        adiis.space = cc.diis_space
    else:
        adiis = lambda t1,t2,*args: (t1,t2)

    conv = False
    for istep in range(max_cycle):
        foo, fov, fvv = make_inter_F(cc, t1, t2, eris)
        t2new = update_amp_t2(cc, t1, t2, eris, foo, fov, fvv)
        normt = numpy.linalg.norm(t2new-t2)
        t1, t2 = t1, t2new
        t1new = t2new = None
        if cc.diis:
            t1, t2 = cc.diis(t1, t2, istep, normt, eccsd-eold, adiis)
        eold, eccsd = eccsd, cc.energy(t1, t2, eris)
        log.info('istep = %d  E(CCD) = %.15g  dE = %.9g  norm(t2) = %.6g',
                 istep, eccsd, eccsd - eold, normt)
        if abs(eccsd-eold) < tol and normt < tolnormt:
            conv = True
            break
    return conv, eccsd, t1, t2


def make_inter_F(cc, t1, t2, eris):
    nocc, nvir = t1.shape
    fock = eris.fock

    tau = t2 
    theta = tau*2.0 - tau.transpose(2,1,0,3)

    foo = fock[:nocc,:nocc].copy()
    foo[range(nocc),range(nocc)] = 0
    g2 = 2.0*numpy.asarray(eris.ovoo)
    g2 -= numpy.asarray(eris.ovoo).transpose(2,1,0,3)
    foo += einsum('iakb,jakb->ij', eris.ovov, theta)
    g2 = None

    fov = fock[:nocc,nocc:]
    g2 = None

    fvv = fock[nocc:,nocc:].copy()
    fvv[range(nvir),range(nvir)] = 0
    g2 = 2.0*numpy.asarray(eris.ovvv)
    g2 -= numpy.asarray(eris.ovvv).transpose(0,3,2,1)
    fvv -= einsum('icja,icjb->ab', theta, eris.ovov)
    return foo, fov, fvv

def update_amp_t2(cc, t1, t2, eris, foo, fov, fvv):
    nocc, nvir = t1.shape
    fock = eris.fock
    t2new = numpy.copy(eris.ovov)

    ft_ij = foo 
    ft_ab = fvv 
    tmp1 = einsum('bc,iajc->iajb', ft_ab, t2)
    tmp2 = einsum('kj,kbia->iajb', ft_ij, t2)
    tw = tmp1 - tmp2
    tmp1 = tmp2 = None

    theta = t2*2.0 - t2.transpose(2,1,0,3)
    tau = theta

    wovOV = numpy.copy(eris.ovov)
    wovOV += 0.5*einsum('jakc,ibkc->ibja', tau, eris.ovov)
    wovOV -= 0.5*einsum('jakc,ickb->ibja', t2, eris.ovov)

    tau = 0.5*t2
    woVoV = -numpy.asarray(eris.oovv).transpose(1,2,0,3)
    woVoV += einsum('jcka,ickb->ibja', tau, eris.ovov)
    tw += einsum('iakc,kcjb->iajb', t2, woVoV)
    tw += einsum('icka,kcjb->ibja', t2, woVoV)
    tw += einsum('iakc,kcjb->iajb', theta, wovOV)
    tau = theta = None

    t2new += tw + tw.transpose(2,3,0,1)
    tmp1 = tw = None
    tau = t2 
    tmp1 = einsum('la,jaik->jlik', t1, eris.ovoo)
    tw = eris.oooo + tmp1 + tmp1.transpose(2,3,0,1)
    tw += einsum('iajb,kalb->ikjl', eris.ovov, tau)
    t2new += einsum('kilj,kalb->iajb', tw, tau)
    t2new += cc.add_wvVvV(t1, t2, eris)

    mo_e = fock.diagonal()
    eia = (mo_e[:nocc,None] - mo_e[None,nocc:]).reshape(-1)
    t2new /= (eia[:,None] + eia[None,:]).reshape(nocc,nvir,nocc,nvir)
    return t2new


def energy(cc, t1, t2, eris):
    tau = t2
    theta = tau*2.0 - tau.transpose(2,1,0,3)
    e = numpy.einsum('iajb,iajb', eris.ovov, theta)
    return e

def get_nocc(mycc):
    if mycc._nocc is not None:
        return mycc._nocc
    elif isinstance(mycc.frozen, (int, numpy.integer)):
        nocc = int(mycc.mo_occ.sum()) // 2 - mycc.frozen
        assert(nocc > 0)
        return nocc
    else:
        occ_idx = mycc.mo_occ > 0
        occ_idx[list(mycc.frozen)] = False
        return numpy.count_nonzero(occ_idx)

def get_nmo(mycc):
    if mycc._nmo is not None:
        return mycc._nmo
    elif isinstance(mycc.frozen, (int, numpy.integer)):
        return len(mycc.mo_occ) - mycc.frozen
    else:
        return len(mycc.mo_occ) - len(mycc.frozen)


class CCD(lib.StreamObject):
    def __init__(self, mf, frozen=0, mo_coeff=None, mo_occ=None):
        if mo_coeff  is None: mo_coeff  = mf.mo_coeff
        if mo_occ    is None: mo_occ    = mf.mo_occ

        self.mol = mf.mol
        self._scf = mf
        self.verbose = self.mol.verbose
        self.stdout = self.mol.stdout
        self.max_memory = mf.max_memory

        self.max_cycle = 50
        self.conv_tol = 1e-7
        self.conv_tol_normt = 1e-5
        self.diis_space = 6
        self.diis_file = None
        self.diis_start_cycle = 1
        self.diis_start_energy_diff = 1e-2

        self.frozen = frozen

##################################################
# don't modify the following attributes, they are not input options
        self.mo_coeff = mo_coeff
        self.mo_occ = mo_occ
        self.emp2 = None
        self.e_corr = None
        self.converged = False
        self._nocc = None
        self._nmo = None
        self.t1 = None
        self.t2 = None

        self._keys = set(self.__dict__.keys())

    def dump_flags(self):
        log = logger.Logger(self.stdout, self.verbose)
        log.info('')
        log.info('******** %s flags ********', self.__class__)
        nocc = self.nocc
        nvir = self.nmo - nocc
        log.info('CCD nocc = %d, nvir = %d', nocc, nvir)
        if self.frozen is not 0:
            log.info('frozen orbitals %s', str(self.frozen))
        log.info('max_cycle = %d', self.max_cycle)
        log.info('conv_tol = %g', self.conv_tol)
        log.info('conv_tol_normt = %s', self.conv_tol_normt)
        log.info('diis_space = %d', self.diis_space)
        log.info('diis_start_cycle = %d', self.diis_start_cycle)
        log.info('diis_start_energy_diff = %g', self.diis_start_energy_diff)
        log.info('max_memory %d MB (current use %d MB)',
                 self.max_memory, lib.current_memory()[0])
        return self

    @property
    def ecc(self):
       return self.e_corr

    @property
    def e_tot(self):
        return self.e_corr + self._scf.e_tot

    nocc = property(get_nocc)
    @nocc.setter
    def nocc(self, n):
        self._nocc = n

    nmo = property(get_nmo)
    @nmo.setter
    def nmo(self, n):
        self._nmo = n

    get_nocc = get_nocc
    get_nmo = get_nmo

    def get_init_guess(self, eris=None):
        if eris is None: eris = self.ao2mo(self.mo_coeff)
        return self.init_amps(eris)
    def init_amps(self, eris):
        nocc = self.nocc
        nvir = self.nmo - nocc
        #nvir = mo_e.size - nocc
        mo_e = eris.fock.diagonal()
        eia = mo_e[:nocc,None] - mo_e[None,nocc:]
        t2 = numpy.empty((nocc,nvir,nocc,nvir))
        self.emp2 = 0
        for i in range(nocc):
            gi = eris.ovov[i]
            t2[i] = gi/lib.direct_sum('a,jb->ajb', eia[i], eia)
            theta = gi*2 - gi.transpose(2,1,0)
            self.emp2 += numpy.einsum('ajb,ajb->', t2[i], theta)
        lib.logger.info(self, 'Init t2, MP2 energy = %.15g', self.emp2)
        t1 = eris.fock[:nocc,nocc:] / eia
        t1[:,:] = 0.0
        return self.emp2, t1, t2

    energy = energy

#   def kernel(self, t1=None, t2=None):
#       eris = self.ao2mo()
#       self.dump_flags()
#       self._conv, self.ecc, self.t1, self.t2 = \
#               kernel(self, eris, t1, t2, max_cycle=self.max_cycle,
#                      tol=self.conv_tol, tolnormt=self.conv_tol_normt,
#                      verbose=self.verbose)
#       return self.ecc, self.t1, self.t2

    def kernel(self, t1=None, t2=None, eris=None):
        return self.ccd(t1, t2, eris)
    def ccd(self, t1=None, t2=None, eris=None):
        if self.verbose >= logger.WARN:
            self.check_sanity()
        self.dump_flags()

        if eris is None:
            eris = self.ao2mo()
        self.converged, self.e_corr, self.t1, self.t2 = \
                kernel(self, eris, t1, t2, max_cycle=self.max_cycle,
                       tol=self.conv_tol, tolnormt=self.conv_tol_normt,
                       verbose=self.verbose)
        if self.converged:
            logger.info(self, 'CCD converged')
        else:
            logger.note(self, 'CCSD not converged')
        if self._scf.e_tot == 0:
            logger.note(self, 'E_corr = %.16g', self.e_corr)
        else:
            logger.note(self, 'E(CCD) = %.16g  E_corr = %.16g',
                        self.e_tot, self.e_corr)
        return self.e_corr, self.t1, self.t2

    def ao2mo(self):
        return _ERIS(self)

    def add_wvVvV(self, t1, t2, eris):
        tau = t2 
        return numpy.einsum('icjd,acbd->iajb', tau, eris.vvvv)

    def diis(self, t1, t2, istep, normt, de, adiis):
        if (istep > self.diis_start_cycle and
            abs(de) < self.diis_start_energy_diff):
            t1t2 = numpy.hstack((t1.ravel(),t2.ravel()))
            t1t2 = adiis.update(t1t2)
            t1 = t1t2[:t1.size].reshape(t1.shape)
            t2 = t1t2[t1.size:].reshape(t2.shape)
            logger.debug(self, 'DIIS for step %d', istep)
        return t1, t2

CC = CCD

class _ERIS:
    def __init__(self, cc, mo_coeff=None):
        if mo_coeff is None:
            self.mo_coeff = mo_coeff = _mo_without_core(cc, cc.mo_coeff)
        else:
            self.mo_coeff = mo_coeff = _mo_without_core(cc, mo_coeff)
# Note: Always recompute the fock matrix because cc._scf.mo_energy may not be
# the eigenvalue of Fock matrix (eg when level shift is used in the scf object
# and the scf does not converge, mo_energy has the contribution of level shift).
        dm = cc._scf.make_rdm1(cc.mo_coeff, cc.mo_occ)
        fockao = cc._scf.get_hcore() + cc._scf.get_veff(cc.mol, dm)
        self.fock = reduce(numpy.dot, (mo_coeff.T, fockao, mo_coeff))

        nocc = cc.nocc
        nmo = cc.nmo
        nvir = nmo - nocc

        eri1 = ao2mo.incore.full(cc._scf._eri, mo_coeff)
        eri1 = ao2mo.restore(1, eri1, nmo)
        self.oooo = eri1[:nocc,:nocc,:nocc,:nocc].copy()
        self.ooov = eri1[:nocc,:nocc,:nocc,nocc:].copy()
        self.ovoo = eri1[:nocc,nocc:,:nocc,:nocc].copy()
        self.oovv = eri1[:nocc,:nocc,nocc:,nocc:].copy()
        self.ovov = eri1[:nocc,nocc:,:nocc,nocc:].copy()
        self.ovvv = eri1[:nocc,nocc:,nocc:,nocc:].copy()
        self.vvvv = eri1[nocc:,nocc:,nocc:,nocc:].copy()

def get_moidx(cc):
    moidx = numpy.ones(cc.mo_occ.size, dtype=numpy.bool)
    if isinstance(cc.frozen, (int, numpy.integer)):
        moidx[:cc.frozen] = False
    elif len(cc.frozen) > 0:
        moidx[list(cc.frozen)] = False
    return moidx
def _mo_without_core(cc, mo):
    return mo[:,get_moidx(cc)]

if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf, cc

    mol = gto.Mole()
    mol.verbose = 4
    mol.atom = '''
    O
    H 1 1.1
    H 1 1.1 2 104
    '''
    mol.basis = '6-31g'
    mol.build()

    rhf = scf.RHF(mol)
    rhf.scf() 

    mcc = CCD(rhf)
    mcc.frozen = 0
    mcc.kernel()

    mycc = cc.CCSD(rhf)
    mycc.frozen = 0
    old_update_amps = mycc.update_amps
    def update_amps(t1, t2, eris):
        t1, t2 = old_update_amps(t1, t2, eris)
        return t1*0, t2
    mycc.update_amps = update_amps
    mycc.kernel()
    
    print('Ref CCD correlation energy', mycc.e_corr)

