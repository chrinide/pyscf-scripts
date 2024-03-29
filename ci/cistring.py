#!/usr/bin/env python

import numpy
import math

def cre_des_sign(p, q, string):
    nset = len(string)
    pg, pb = p//64, p%64
    qg, qb = q//64, q%64

    if pg > qg:
        n1 = 0
        for i in range(nset-pg, nset-qg-1):
            n1 += bin(string[i]).count('1')
        n1 += bin(string[-1-pg] & numpy.uint64((1<<pb) - 1)).count('1')
        n1 += string[-1-qg] >> numpy.uint64(qb+1)
    elif pg < qg:
        n1 = 0
        for i in range(nset-qg, nset-pg-1):
            n1 += bin(string[i]).count('1')
        n1 += bin(string[-1-qg] & numpy.uint64((1<<qb) - 1)).count('1')
        n1 += string[-1-pg] >> numpy.uint64(pb+1)
    else:
        if p > q:
            mask = numpy.uint64((1 << pb) - (1 << (qb+1)))
        else:
            mask = numpy.uint64((1 << qb) - (1 << (pb+1)))
        n1 = bin(string[-1-pg]&mask).count('1')

    if n1 % 2:
        return -1
    else:
        return 1

def str_diff(string0, string1):
    des_string0 = []
    cre_string0 = []
    nset = len(string0)
    off = 0
    for i in reversed(range(nset)):
        df = string0[i] ^ string1[i]
        des_string0.extend([x+off for x in find1(df & string0[i])])
        cre_string0.extend([x+off for x in find1(df & string1[i])])
        off += 64
    return des_string0, cre_string0

def excitation_level(string, nelec=None):
    nset = len(string)
    if nelec is None:
        nelec = 0
        for i in range(nset):
            nelec += bin(string[i]).count('1')

    g, b = nelec//64, nelec%64
    tn = nelec - bin(string[-1-g])[-b:].count('1')
    for s in string[nset-g:]:
        tn -= bin(s).count('1')
    return tn

def find1(s):
    return [i for i,x in enumerate(bin(s)[2:][::-1]) if x is '1']

def toggle_bit(s, place):
    nset = len(s)
    g, b = place//64, place%64
    s[-1-g] ^= numpy.uint64(1<<b)
    return s

def str2orblst(string, norb):
    occ = []
    vir = []
    nset = len(string)
    off = 0
    for k in reversed(range(nset)):
        s = string[k]
        occ.extend([x+off for x in find1(s)])
        for i in range(0, min(64, norb-off)): 
            if not (s & numpy.uint64(1<<i)):
                vir.append(i+off)
        off += 64
    return occ, vir

def orblst2str(lst, norb):
    nset = (norb+63) // 64
    string = numpy.zeros(nset, dtype=numpy.uint64)
    for i in lst:
        toggle_bit(string, i)
    return string

def gen_strings4orblist(orb_list, nelec):
    orb_list = list(orb_list)
    if len(orb_list) > 63:
        raise RuntimeError('''
You see this error message because we only support up to 64 orbitals''')

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

def num_strings(n, m):
    if m < 0 or m > n:
        return 0
    else:
        return math.factorial(n) // (math.factorial(n-m)*math.factorial(m))

class _SCIvector(numpy.ndarray):
    def __array_finalize__(self, obj):
        self._strs = getattr(obj, '_strs', None)

def as_SCIvector(civec, ci_strs):
    civec = civec.view(_SCIvector)
    civec._strs = ci_strs
    return civec

def as_SCIvector_if_not(civec, ci_strs):
    if not hasattr(civec, '_strs'):
        civec = as_SCIvector(civec, ci_strs)
    return civec

# TODO: fast loop over alpha/beta, add excitation operators
def gen_full_space(orb_list, nelec):

    if len(orb_list) > 63:
        raise RuntimeError('''
You see this error message because we only support up to 64 orbitals''')
    
    neleca = nelec[0]
    nelecb = nelec[1]
    stringa = gen_strings4orblist(orb_list, neleca)
    stringb = gen_strings4orblist(orb_list, nelecb)
    ndets = stringa.shape[0]*stringb.shape[0]
    dets = numpy.zeros((ndets,2), dtype=numpy.uint64)
    #print ndets
    k = 0
    for i in range(stringa.shape[0]):
        for j in range(stringb.shape[0]):
            dets[k,0] = stringa[i]
            dets[k,1] = stringb[j]
            k += 1
    return dets

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
    for a in range(nocc, norb):
        for i in reversed(range(nocc)):
            str1 = hf_str ^ (1 << i) | (1 << a)
            strs.append(str1)
    return numpy.asarray(strs)

if __name__ == '__main__':
    numpy.random.seed(3)
    nelec = (1,1)
    norb = 2

    print "From HF"
    hf_str = numpy.hstack([orblst2str(range(nelec[0]), norb), \
                           orblst2str(range(nelec[1]), norb)]).reshape(1,-1)
    ndets = hf_str.shape[0] 
    ci0 = [as_SCIvector(numpy.ones(ndets), hf_str)]
    print "Number of roots", len(ci0) 
    print "0 root coeffs", ci0[0]
    strs = ci0[0]._strs
    print "Number of strings", strs.shape
    print "String", bin(strs[0,0]), bin(strs[0,1])

    print "Full space"
    dets = gen_full_space(range(norb), nelec) 
    ndets = dets.shape[0]
    ci1 = [as_SCIvector(numpy.zeros(ndets), dets)]
    ci1[0][0] = 1.0
    print "Number of roots", len(ci1) 
    print "0 root coeffs", ci1[0]
    strs = ci1[0]._strs
    print "Number of strings", strs.shape
    print "HF String", bin(strs[0,0]), bin(strs[0,1])
    for i in range(ndets):
        print "Det",i,"alpha",bin(strs[i,0]),"beta",bin(strs[i,1])
    
    print "T1"
    nelec = (5,5)
    norb = 14
    detsa = tn_strs(norb, nelec[0], 1)
    detsb = tn_strs(norb, nelec[1], 1)
    ndets = detsa.shape[0]*detsb.shape[0]
    dets = numpy.zeros((ndets,2), dtype=numpy.uint64)
    k = 0
    for i in range(detsa.shape[0]):
        for j in range(detsb.shape[0]):
            dets[k,0] = detsa[i]
            dets[k,1] = detsb[j]
            k += 1
    ci1 = [as_SCIvector(numpy.zeros(ndets), dets)]
    ci1[0][0] = 1.0
    print "Number of roots", len(ci1) 
    print "0 root coeffs", ci1[0]
    strs = ci1[0]._strs
    print "Number of strings", strs.shape
    for i in range(ndets):
        print "Det",i,"alpha",bin(strs[i,0]),"beta",bin(strs[i,1])
    print "T2"
    detsa = tn_strs(norb, nelec[0], 2)
    detsb = tn_strs(norb, nelec[1], 2)
    ndets = detsa.shape[0]*detsb.shape[0]
    dets = numpy.zeros((ndets,2), dtype=numpy.uint64)
    k = 0
    for i in range(detsa.shape[0]):
        for j in range(detsb.shape[0]):
            dets[k,0] = detsa[i]
            dets[k,1] = detsb[j]
            k += 1
    ci1 = [as_SCIvector(numpy.zeros(ndets), dets)]
    ci1[0][0] = 1.0
    print "Number of roots", len(ci1) 
    print "0 root coeffs", ci1[0]
    strs = ci1[0]._strs
    print "Number of strings", strs.shape
    for i in range(ndets):
        print "Det",i,"alpha",bin(strs[i,0]),"beta",bin(strs[i,1])

    dets = t1_strs(norb, nelec[0])
    for i in range(dets.shape[0]):
        print "Det",i,"det",bin(dets[i])
