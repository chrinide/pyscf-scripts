#!/usr/bin/env python

import sys
import math
import numpy
import ctypes

def tn_strs(norb, nelec, n):
    '''Generate strings for Tn amplitudes.  Eg n=1 (T1) has nvir*nocc strings,
    n=2 (T2) has nvir*(nvir-1)/2 * nocc*(nocc-1)/2 strings.
    '''
    if nelec < n or norb-nelec < n:
        return numpy.zeros(0, dtype=int)
    occs_allow = numpy.asarray(gen_strings4orblist(range(nelec), n)[::-1])
    virs_allow = numpy.asarray(gen_strings4orblist(range(nelec,norb), n))
    hf_str = int('1'*nelec, 2)
    tns = (hf_str | virs_allow.reshape(-1,1)) ^ occs_allow
    return tns.ravel()

def t1_strs(norb, nelec):
    '''CIS dets'''
    nocc = nelec
    hf_str = int('1'*nocc, 2)
    strs = []
    signs = []
    for a in range(nocc, norb):
        for i in reversed(range(nocc)):
            str1 = hf_str ^ (1 << i) | (1 << a)
            strs.append(str1)
            signs.append(cre_des_sign(a, i, hf_str))
    return strs
    
def str2orbidx(string):
    bstring = bin(string)
    return [i for i,s in enumerate(bstring[::-1]) if s == '1']

def wfnstr2orbidx(string):
    bstring = bin(string)
    return [i+1 for i,s in enumerate(bstring[::-1]) if s == '1']

if __name__ == '__main__':

    #neleca = 1
    #nmo = 4
    #nalpha = num_strings(nmoa,neleca)
    #alpha = gen_strings4orblist(range(nmoa), neleca)
    #for i in range(nalpha):
    #   print bin(alpha[i])
    #occlsta = _gen_occslst(range(nmoa), neleca)
    #print occlsta
    #print gen_cre_str_index_o0(range(nmoa), neleca)
    
    norb = 4
    neleca = 1
    nelecb = 1
    strsa = t1_strs(norb, neleca)#, 1) 
    stradic = dict(zip(strsa,range(strsa.__len__())))
    strsb = t1_strs(norb, nelecb)#, 1) 
    strbdic = dict(zip(strsb,range(strsb.__len__())))
    na = len(stradic)
    nb = len(strbdic)
    ndet = na*nb
    print na,nb,na*nb

    for i in range(na):
        for j in range(nb):
            idxa = ['%3d' % x for x in str2orbidx(strsa[i])]
            idxb = ['%3d' % (-x) for x in str2orbidx(strsb[j])]
            print ' '.join(idxa), ' '.join(idxb)

    norb = 4
    neleca = 1
    nelecb = 1
    strsa = gen_strings4orblist(range(norb), neleca)
    stradic = dict(zip(strsa,range(strsa.__len__())))
    strsb = gen_strings4orblist(range(norb), nelecb)
    strbdic = dict(zip(strsb,range(strsb.__len__())))
    na = len(stradic)
    nb = len(strbdic)
    ndet = na*nb
    print na,nb,na*nb

    for i in range(na):
        for j in range(nb):
            idxa = ['%3d' % x for x in str2orbidx(strsa[i])]
            idxb = ['%3d' % (-x) for x in str2orbidx(strsb[j])]
            print ' '.join(idxa), ' '.join(idxb)

    from pyscf import gto, scf, ao2mo, fci, mcscf

    mol = gto.Mole()
    mol.basis = '3-21g'
    mol.atom = '''
    H  0.0000  0.0000  0.0000
    H  0.0000  0.0000  3.7500
    '''
    mol.verbose = 0
    mol.spin = 0
    mol.charge = 0
    mol.symmetry = 1
    mol.build()

    mf = scf.RHF(mol)
    ehf = mf.kernel()
    print ehf

    nao, nmo = mf.mo_coeff.shape
    h1 = reduce(numpy.dot, (mf.mo_coeff[:,:nmo].T, mf.get_hcore(), mf.mo_coeff[:,:nmo]))
    eri_mo = ao2mo.kernel(mf._eri, mf.mo_coeff[:,:nmo], compact=False)
    eri_mo = eri_mo.reshape(nmo,nmo,nmo,nmo)

