#!/usr/bin/env python

import os
import sys
import math
import numpy
import ctypes
import signal
from functools import reduce

from pyscf import lib

signal.signal(signal.SIGINT, signal.SIG_DFL)

# For code compatiblity in python-2 and python-3
if sys.version_info >= (3,):
    unicode = str

_loaderpath = os.path.dirname(__file__)
libci = numpy.ctypeslib.load_library('libci.so', _loaderpath)

def find1(s):
    return [i for i,x in enumerate(bin(s)[2:][::-1]) if x is '1']

def str2orblst(string, norb):
    occ = []
    vir = []
    occ.extend([x for x in find1(string)])
    for i in range(norb): 
        if not (string & 1<<i):
            vir.append(i)
    return occ, vir

def num_strings(n, m):
    if m < 0 or m > n:
        return 0
    else:
        return math.factorial(n) // (math.factorial(n-m)*math.factorial(m))

def str_diff(string0, string1):
    des_string0 = []
    cre_string0 = []
    df = string0 ^ string1
    des_string0.extend([x for x in find1(df & string0)])
    cre_string0.extend([x for x in find1(df & string1)])
    return des_string0, cre_string0    

def cre_des_sign(p, q, string0):
    if p == q:
        return 1
    else:
        if (string0 & (1<<p)) or (not (string0 & (1<<q))):
            return 0
        elif p > q:
            mask = (1 << p) - (1 << (q+1))
        else:
            mask = (1 << q) - (1 << (p+1))
        return (-1) ** bin(string0 & mask).count('1')

def print_dets(self,strs):
    ndets = strs.shape[0]
    lib.logger.info(self, '*** Printing list of determinants')
    lib.logger.info(self, 'Number of determinants: %s', ndets)
    for i in range(ndets):
        lib.logger.info(self,'Det %d %s %s' % (i,bin(strs[i,0]),bin(strs[i,1])))
    return self

def make_strings(self,orb_list,nelec):
    orb_list = list(orb_list)
    assert(nelec >= 0)
    if nelec == 0:
        return numpy.asarray([0], dtype=numpy.int64)
    elif nelec > len(orb_list):
        return numpy.asarray([], dtype=numpy.int64)
    def gen_str_iter(orb_list, nelec):
        if nelec == 1:
            res = [(1<<i) for i in orb_list]
        elif nelec >= len(orb_list):
            n = 0
            for i in orb_list:
                n = n | (1<<i)
            res = [n]
        else:
            restorb = orb_list[:-1]
            thisorb = 1 << orb_list[-1]
            res = gen_str_iter(restorb, nelec)
            for n in gen_str_iter(restorb, nelec-1):
                res.append(n | thisorb)
        return res
    strings = gen_str_iter(orb_list, nelec)
    assert(strings.__len__() == num_strings(len(orb_list),nelec))
    return numpy.asarray(strings, dtype=numpy.int64)

def make_hdiag(self,h1e,h2e,h,strs):
    ndets = strs.shape[0]
    norb = h1e.shape[0]//2
    diagj = numpy.einsum('iijj->ij', h2e)
    diagk = numpy.einsum('ijji->ij', h2e)
    for idet in range(ndets):
        strp = strs[idet,0]
        strn = strs[idet,1]
        occp = str2orblst(strp, norb)[0]
        occn = str2orblst(strn, norb)[0]
        e1 = h1e[occp,occp].sum() + h1e[occn,occn].sum()
        e2 = diagj[occp][:,occp].sum() + diagj[occp][:,occn].sum() \
           + diagj[occn][:,occp].sum() + diagj[occn][:,occn].sum() \
           - diagk[occp][:,occp].sum() - diagk[occn][:,occn].sum()
        h[idet,idet] = e1 + e2*0.5
    return h

def make_hoffdiag(self,h1e,h2e,h,strs):
    ndets = strs.shape[0]
    norb = h1e.shape[0]//2
    for ip in range(ndets):
        for jp in range(ip):
            strip, strin = strs[ip,0], strs[ip,1]
            strjp, strjn = strs[jp,0], strs[jp,1]
            desp, crep = str_diff(strip, strjp)
            if len(desp) > 2:
                continue
            desn, cren = str_diff(strin, strjn)
            if len(desn) + len(desp) > 2:
                continue
            if len(desp) + len(desn) == 1:
# p->p
                if len(desn) == 0:
                    i,a = desp[0], crep[0]
                    occp = str2orblst(strip, norb)[0]
                    occn = str2orblst(strin, norb)[0]
                    fai = h1e[a,i]
                    for k in occp:
                        fai += h2e[k,k,a,i] - h2e[k,i,a,k]
                    for k in occn:
                        fai += h2e[k,k,a,i]
                    sign = cre_des_sign(a, i, strip)
                    h[ip,jp] = sign * fai
                    h[jp,ip] = h[ip,jp].conj()
# n->n
                elif len(desp) == 0:
                    i,a = desn[0], cren[0]
                    occp = str2orblst(strip, norb)[0]
                    occn = str2orblst(strin, norb)[0]
                    fai = h1e[a,i]
                    for k in occn:
                        fai += h2e[k,k,a,i] - h2e[k,i,a,k]
                    for k in occp:
                        fai += h2e[k,k,a,i]
                    sign = cre_des_sign(a, i, strin)
                    h[ip,jp] = sign * fai
                    h[jp,ip] = h[ip,jp].conj()
            else:
# p,p->p,p
                if len(desn) == 0:
                    i,j = desp
                    a,b = crep
# 6 conditions for i,j,a,b
# --++, ++--, -+-+, +-+-, -++-, +--+ 
                    if a > j or i > b:
# condition --++, ++--
                        v = h2e[a,j,b,i] - h2e[a,i,b,j]
                        sign = cre_des_sign(b, i, strip)
                        sign*= cre_des_sign(a, j, strip)
                    else:
# condition -+-+, +-+-, -++-, +--+ 
                        v = h2e[a,i,b,j] - h2e[a,j,b,i]
                        sign = cre_des_sign(b, j, strip)
                        sign*= cre_des_sign(a, i, strip)
                    h[ip,jp] = sign * v
                    h[jp,ip] = h[ip,jp].conj()
# n,n->n,n
                elif len(desp) == 0:
                    i,j = desn
                    a,b = cren
                    if a > j or i > b:
                        v = h2e[a,j,b,i] - h2e[a,i,b,j]
                        sign = cre_des_sign(b, i, strin)
                        sign*= cre_des_sign(a, j, strin)
                    else:
                        v = h2e[a,i,b,j] - h2e[a,j,b,i]
                        sign = cre_des_sign(b, j, strin)
                        sign*= cre_des_sign(a, i, strin)
                    h[ip,jp] = sign * v
                    h[jp,ip] = h[ip,jp].conj()
# p,n->p,n
                else:
                    i,a = desp[0], crep[0]
                    j,b = desn[0], cren[0]
                    v = h2e[a,i,b,j]
                    sign = cre_des_sign(a, i, strip)
                    sign*= cre_des_sign(b, j, strin)
                    h[ip,jp] = sign * v
                    h[jp,ip] = h[ip,jp].conj()
    return h

if __name__ == '__main__':
    import time
    from pyscf import gto, scf, x2c, ao2mo
    from pyscf.tools.dump_mat import dump_tri

    name = 'slater'
    mol = gto.Mole()
    mol.basis = 'sto-6g'
    mol.atom = '''
    H  0.0 0.0 0.00
    H  0.0 0.0 9.75
    '''
    mol.verbose = 4
# In case of no nelec%2 reduce to cation and then change in the CI
# This ensure Kramers time reversal symmetry for orbitals if x2c.RHF
# is used, symmetry is imposed on density matrix, then one only has
# to classifie it by mo_energy and time operator
    mol.spin = 0
    mol.charge = 0
    mol.symmetry = 0
    mol.build()

    mf = x2c.RHF(mol)
    mf.chkfile = name+'.chk'
    mf.with_x2c.basis = 'unc-ano'
    dm = mf.get_init_guess() + 0.1j
    ehf = mf.kernel(dm)
    coeff = mf.mo_coeff

    lib.logger.TIMER_LEVEL = 3

    ncore = 0
    norb = 4
    nelec = mol.nelectron - ncore
    e_core = mol.energy_nuc() 
    nao, nmo = coeff.shape
    nvir = nmo - ncore - norb
    lib.logger.info(mf, '\n *** A simple relativistic CI module')
    lib.logger.info(mf, 'Number of occupied core 2C spinors %s', ncore)
    lib.logger.info(mf, 'Number of virtual core 2C spinors %s', nvir)
    lib.logger.info(mf, 'Number of electrons to be correlated %s', nelec)
    lib.logger.info(mf, 'Number of 2C spinors to be correlated %s', norb)
    if (norb > 64):
        raise RuntimeError('''Only support up to 64 orbitals''')

    hcore = mf.get_hcore()
    corevhf = numpy.zeros((nao,nao), dtype=numpy.complex128)
    if (ncore != 0):
        core_idx = numpy.arange(ncore)
        core_dm = numpy.dot(coeff[:, core_idx], coeff[:, core_idx].conj().T)
        e_core += numpy.einsum('ij,ji->', core_dm, hcore)
        corevhf = mf.get_veff(mol, core_dm)
        e_core += numpy.einsum('ij,ji->', core_dm, corevhf)*0.5
        e_core = e_core.real

    sequence = numpy.arange(norb)
    ncorep = ncore//2
    ncoren = ncore//2
    # Now only check with manual selecction after coeff check
    p = [0,2]
    n = [1,3]
    cip_idx = p
    cin_idx = n
    nelecp = nelec//2
    nelecn = nelec//2
    strsp = make_strings(mf,cip_idx,nelecp) 
    strsn = make_strings(mf,cin_idx,nelecn) 
    ndetsp = strsp.shape[0]
    ndetsn = strsn.shape[0]
    ndets = ndetsp*ndetsn
    lib.logger.info(mf, 'Number of dets in civec %s', ndets)
    strs = numpy.zeros((ndets,2), dtype=numpy.int64)
    k = 0
    for i in range(ndetsp):
        for j in range(ndetsn):
            strs[k,0] = strsp[i]
            strs[k,1] = strsn[j]
            k += 1
    print_dets(mf,strs) 

    ci_idx = ncore + numpy.arange(norb)
    h1e = reduce(numpy.dot, (coeff[:, ci_idx].conj().T, hcore+corevhf, coeff[:,ci_idx]))
    eri_mo = ao2mo.kernel(mol, coeff[:, ci_idx], compact=False, intor='int2e_spinor')
    eri_mo = eri_mo.reshape(norb,norb,norb,norb)

    h = numpy.zeros((ndets,ndets), dtype=numpy.complex128)
    h = make_hdiag(mf,h1e,eri_mo,h,strs) 
    h = make_hoffdiag(mf,h1e,eri_mo,h,strs) 
    dump_tri(mf.stdout,h,ncol=15,digits=4)
    print h[0,0]+e_core
    e, c = numpy.linalg.eigh(h)
    e += e_core
    lib.logger.info(mf, 'Core energy %s', e_core)
    lib.logger.info(mf, 'Ground state energy %s', e[0])
    lib.logger.info(mf, 'Correlation energy %s', (e[0]-ehf))
    norm = numpy.einsum('i,i->',c[:,0].conj(),c[:,0])
    lib.logger.info(mf, 'Norm of ground state civec %s', norm)
    lib.logger.info(mf, 'CI ground state civec %s', c[:,0])

    from pyscf import mcscf
    myhf = scf.RHF(mol).x2c()
    myhf.verbose = 0
    myhf.kernel()
    mycas = mcscf.CASCI(myhf, 2, 2)
    mycas.verbose = 4
    mycas.kernel()
