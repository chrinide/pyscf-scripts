#!/usr/bin/env python

import os
import sys
import math
import numpy
import ctypes
import signal
from functools import reduce

from pyscf import scf, lib, ao2mo

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

def orblst2str(lst):
    string = numpy.uint64(0)
    for idx in lst:
        string ^= numpy.uint64(1<<idx)
    return string

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
    logger.info(self, '*** Printing list of determinants')
    logger.info(self, 'Number of determinants: %s', ndets)
    for i in range(ndets):
        logger.info(self,'Det %d %s' % (i,bin(strs[i])))
    return self

def make_strings(self,orb_list,nelec):
    '''Generate string from the given orbital list.

    Returns:
        list of int64.  One int64 element represents one string in binary format.
        The binary format takes the convention that the one bit stands for one
        orbital, bit-1 means occupied and bit-0 means unoccupied.  The lowest
        (right-most) bit corresponds to the lowest orbital in the orb_list.

    Exampels:

    >>> [bin(x) for x in make_strings((0,1,2,3),2)]
    [0b11, 0b101, 0b110, 0b1001, 0b1010, 0b1100]
    >>> [bin(x) for x in make_strings((3,1,0,2),2)]
    [0b1010, 0b1001, 0b11, 0b1100, 0b110, 0b101]
    '''
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

# TODO: When large number of dets it should be passed to C
def make_hdiag(self,h1e,h2e,h,strs):
    ndets = strs.shape[0]
    norb = h1e.shape[0]
    diagj = lib.einsum('iijj->ij', h2e)
    diagk = lib.einsum('ijji->ij', h2e)
    for i in range(ndets):
        stri = strs[i]
        occs = str2orblst(stri, norb)[0]
        e1 = h1e[occs,occs].sum()
        e2 = diagj[occs][:,occs].sum() \
           - diagk[occs][:,occs].sum()
        h[i] = e1 + e2*0.5
    return h

def c_make_hdiag(self,h1e,h2e,hdiag,strs,nelec):
    ndets = strs.shape[0]
    norb = h1e.shape[0]
    diagj = lib.einsum('iijj->ij', h2e)
    diagk = lib.einsum('ijji->ij', h2e)
    h1e = numpy.asarray(h1e, order='C')
    diagj = numpy.asarray(diagj, order='C')
    diagk = numpy.asarray(diagk, order='C')
    hdiag = numpy.asarray(hdiag, order='C')
    strs = numpy.asarray(strs, order='C')

    libci.diagonal(h1e.ctypes.data_as(ctypes.c_void_p), 
                   diagj.ctypes.data_as(ctypes.c_void_p), 
                   diagk.ctypes.data_as(ctypes.c_void_p), 
                   ctypes.c_int(norb), 
                   ctypes.c_int(nelec), 
                   strs.ctypes.data_as(ctypes.c_void_p), 
                   ctypes.c_ulonglong(ndets), 
                   hdiag.ctypes.data_as(ctypes.c_void_p)) 

    return hdiag

# TODO: Write in C code to speed up
def make_hoffdiag(self,h1e,h2e,h,strs):
    ndets = strs.shape[0]
    norb = h1e.shape[0]
    for ip in range(ndets):
        for jp in range(ip):
            stri = strs[ip]
            strj = strs[jp]
            des, cre = str_diff(stri, strj)
            if (len(des) == 1):
                i,a = des[0], cre[0]
                occs = str2orblst(stri, norb)[0]
                v = h1e[a,i]
                for k in occs:
                    v += h2e[k,k,a,i] - h2e[k,i,a,k]
                sign = cre_des_sign(a, i, stri)
                h[ip,jp] = sign * v
                h[jp,ip] = h[ip,jp].conj()
            elif (len(des) == 2):
                i,j = des
                a,b = cre
                if a > j or i > b:
                    v = h2e[a,j,b,i] - h2e[a,i,b,j]
                    sign = cre_des_sign(b, i, stri)
                    sign*= cre_des_sign(a, j, stri)
                else:
                    v = h2e[a,i,b,j] - h2e[a,j,b,i]
                    sign = cre_des_sign(b, j, stri)
                    sign*= cre_des_sign(a, i, stri)
                h[ip,jp] = sign * v
                h[jp,ip] = h[ip,jp].conj()
            else:
                continue
    return h

def contract(self,h1e,h2e,hdiag,civec,strs):
    ndets = strs.shape[0]
    norb = h1e.shape[0]
    ci1 = numpy.zeros_like(civec)
    for ip in range(ndets):
        for jp in range(ip):
            stri = strs[ip]
            strj = strs[jp]
            des, cre = str_diff(stri, strj)
            if (len(des) == 1):
                i,a = des[0], cre[0]
                occs = str2orblst(stri, norb)[0]
                v = h1e[a,i]
                for k in occs:
                    v += h2e[k,k,a,i] - h2e[k,i,a,k]
                sign = cre_des_sign(a, i, stri)
                ci1[jp] += sign*v*civec[ip]
                ci1[ip] += (sign*v).conj()*civec[jp]
            elif (len(des) == 2):
                i,j = des
                a,b = cre
                if a > j or i > b:
                    v = h2e[a,j,b,i] - h2e[a,i,b,j]
                    sign = cre_des_sign(b, i, stri)
                    sign*= cre_des_sign(a, j, stri)
                else:
                    v = h2e[a,i,b,j] - h2e[a,j,b,i]
                    sign = cre_des_sign(b, j, stri)
                    sign*= cre_des_sign(a, i, stri)
                ci1[jp] += sign*v*civec[ip]
                ci1[ip] += (sign*v).conj()*civec[jp]
            else:
                continue
        ci1[ip] += hdiag[ip] * civec[ip]
    return ci1

def c_contract(self,h1e,h2e,hdiag,civec,strs,nelec):
    ndets = strs.shape[0]
    norb = h1e.shape[0]
    ci1 = numpy.zeros_like(civec)
    h1e = numpy.asarray(h1e, order='C')
    h2e = numpy.asarray(h2e, order='C')
    strs = numpy.asarray(strs, order='C')
    civec = numpy.asarray(civec, order='C')
    hdiag = numpy.asarray(hdiag, order='C')
    ci1 = numpy.asarray(ci1, order='C')

    libci.contract(h1e.ctypes.data_as(ctypes.c_void_p), 
                   h2e.ctypes.data_as(ctypes.c_void_p), 
                   ctypes.c_int(norb), 
                   ctypes.c_int(nelec), 
                   strs.ctypes.data_as(ctypes.c_void_p), 
                   civec.ctypes.data_as(ctypes.c_void_p), 
                   hdiag.ctypes.data_as(ctypes.c_void_p), 
                   ctypes.c_ulonglong(ndets), 
                   ci1.ctypes.data_as(ctypes.c_void_p))
    return ci1

# TODO: Add core/core-valence contribution
# TODO: to slow write in C
def make_rdm1(self,strs,civec,norb):
    ndets = strs.shape[0]
    rdm1 = numpy.zeros((norb,norb), dtype=numpy.complex128)
    for ip in range(ndets):
        stri = strs[ip]
        occs = str2orblst(stri, norb)[0]
        ci2 = civec[ip] * civec[ip].conj()
        for k in occs:
            rdm1[k,k] += ci2
        for jp in range(ip+1):
            strj = strs[jp]
            des, cre = str_diff(stri, strj)
            if (len(des) == 1):
                i,a = des[0], cre[0]
                ci2 = civec[ip] * civec[jp].conj()
                rdm1[i,a] += ci2
                rdm1[a,i] += rdm1[i,a].conj()
            else:
                continue
    return rdm1

if __name__ == '__main__':
    import time
    from pyscf import gto, scf, x2c, ao2mo
    from pyscf.tools.dump_mat import dump_tri
    mol = gto.Mole()
    mol.basis = 'sto-6g'
    mol.atom = '''
N      0.000000      0.000000      0.000000
N      0.000000      0.000000      1.100000
    '''
#Li      0.000000      0.000000      1.371504
#Li      0.000000      0.000000     -1.371504
    mol.verbose = 4
    mol.spin = 0
    mol.charge = 0
    mol.symmetry = 0
    mol.build()

    mf = x2c.RHF(mol)
    mf.with_x2c.basis = 'unc-ano'
    dm = mf.get_init_guess() + 0.1j
    mf.kernel(dm)

    lib.logger.TIMER_LEVEL = 3

    ncore = 4
    nelec = mol.nelectron - ncore
    e_core = mol.energy_nuc() 
    nao, nmo = mf.mo_coeff.shape
    nvir = 0
    norb = nmo - ncore - nvir
    coeff = mf.mo_coeff
    lib.logger.info(mf, '\n *** A simple relativistic CI module')
    lib.logger.info(mf, 'Number of occupied core 2C spinors %s', ncore)
    lib.logger.info(mf, 'Number of virtual core 2C spinors %s', nvir)
    lib.logger.info(mf, 'Number of electrons to be correlated %s', nelec)
    lib.logger.info(mf, 'Number of 2C spinors to be correlated %s', norb)
    if (norb > 64):
        raise RuntimeError('''Only support up to 64 orbitals''')

    t0 = (time.clock(), time.time())
    hcore = mf.get_hcore()
    corevhf = numpy.zeros((nao,nao), dtype=numpy.complex128)
    if (ncore != 0):
        core_idx = numpy.arange(ncore)
        core_dm = numpy.dot(coeff[:, core_idx], coeff[:, core_idx].conj().T)
        e_core += numpy.einsum('ij,ji->', core_dm, hcore)
        corevhf = mf.get_veff(mol, core_dm)
        e_core += numpy.einsum('ij,ji->', core_dm, corevhf)*0.5
        e_core = e_core.real

    ci_idx = ncore + numpy.arange(norb)
    h1e = reduce(numpy.dot, (coeff[:, ci_idx].conj().T, hcore+corevhf, coeff[:,ci_idx]))
    t5 = (time.clock(), time.time())
    eri_mo = ao2mo.kernel(mol, coeff[:, ci_idx], compact=False, intor='int2e_spinor')
    lib.logger.timer(mf,'ao2mo build', *t5)
    eri_mo = eri_mo.reshape(norb,norb,norb,norb)

    orb_list = list(range(norb))
    t1 = (time.clock(), time.time())
    strs = make_strings(mf,orb_list,nelec) 
    lib.logger.timer(mf,'Strings build', *t1)
    ndets = strs.shape[0]
    lib.logger.info(mf, 'Number of dets in civec %s', ndets)
    #h = numpy.zeros((ndets,ndets), dtype=numpy.complex128)
    hdiag = numpy.zeros(ndets, dtype=numpy.complex128)
    t2 = (time.clock(), time.time())
    hdiag = make_hdiag(mf,h1e,eri_mo,hdiag,strs) 
    lib.logger.timer(mf,'<i|H|i> build', *t2)

    t3 = (time.clock(), time.time())
    #h = make_hoffdiag(mf,h1e,eri_mo,h,strs)
    #lib.logger.timer(mf,'<i|H|j> build', *t3)
    #dump_tri(mf.stdout,h,ncol=15,digits=4)
    #ci0 = numpy.zeros(ndets, dtype=numpy.complex128)
    #ci0[0] = 1.0
    numpy.random.seed()
    ci0 = numpy.random.uniform(low=1e-3,high=1e-2,size=ndets)
    ci0 = ci0.astype(numpy.complex128)
    ci0[0] = 0.99
    ci0 *= 1./numpy.linalg.norm(ci0)
    def hop(c):
        #hc = contract(mf,h1e,eri_mo,hdiag,c,strs)
        hc = c_contract(mf,h1e,eri_mo,hdiag,c,strs,nelec)
        return hc.ravel()
    level_shift = 0.01
    precond = lambda x, e, *args: x/(hdiag-e+level_shift)

    t4 = (time.clock(), time.time())
    #e,c = numpy.linalg.eigh(h)
    nthreads = lib.num_threads()
    conv_tol = 1e-12
    lindep = 1e-14
    max_cycle = 100
    max_space = 12
    lessio = False
    max_memory = 2000
    follow_state = False 
    nroots = 1
    #with lib.with_omp_threads(nthreads):
    e, c = lib.davidson(hop, ci0, precond, tol=conv_tol, lindep=lindep)
                           #max_cycle=max_cycle, max_space=max_space, max_memory=max_memory, 
                           #dot=numpy.dot, nroots=nroots, lessio=lessio, verbose=mf.verbose,
                           #follow_state=follow_state)
    #lib.logger.timer(mf,'<i|H|j> diagonalization', *t4)
    lib.logger.timer(mf,'<i|H|j> Davidson', *t4)
    e += e_core
    lib.logger.info(mf, 'Core energy %s', e_core)
    #lib.logger.info(mf, 'Ground state energy %s', e[0])
    lib.logger.info(mf, 'Ground state energy %s', e)
    #lib.logger.info(mf, 'Ground state civec %s', c[:,0])
    #norm = numpy.einsum('i,i->',c[:,0].conj(),c[:,0])
    norm = numpy.einsum('i,i->',c.conj(),c)
    lib.logger.info(mf, 'Norm of ground state civec %s', norm)
    lib.logger.timer(mf,'CI build', *t0)

    t0 = (time.clock(), time.time())
    #rdm1 = make_rdm1(mf,strs,c[:,0],norb)
    #rdm1 = make_rdm1(mf,strs,c,norb)
    #dump_tri(mf.stdout,rdm1,ncol=15,digits=4)
    #print rdm1
    #nelec = numpy.einsum('ii->', rdm1)
    #lib.logger.info(mf, 'Number of electrons in active space %s', nelec)
    #lib.logger.timer(mf,'1-RDM build', *t0)

    #nelec = mol.nelectron - ncore
    #c_contract(mf,h1e,eri_mo,hdiag,ci0,strs,nelec) 

