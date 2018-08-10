#!/usr/bin/env python

import numpy
import time
import ctypes
from pyscf import lib
from pyscf import ao2mo
from pyscf.lib import logger
from pyscf.fci import cistring
from pyscf.fci import direct_spin1

libhci = lib.load_library('libhci')

def to_fci_wfn(fout, civec, norb, nelec, root=0, ncore=0):
    if isinstance(nelec, (int, numpy.number)):
        nelecb = nelec//2
        neleca = nelec - nelecb
    else:
        neleca, nelecb = nelec
    strsa = cistring.gen_strings4orblist(range(norb), neleca)
    stradic = dict(zip(strsa,range(strsa.__len__())))
    strsb = cistring.gen_strings4orblist(range(norb), nelecb)
    strbdic = dict(zip(strsb,range(strsb.__len__())))
    na = len(stradic)
    nb = len(strbdic)
    ndet = len(civec[root])
    from pyscf import fci
    stringsa = fci.cistring.gen_strings4orblist(range(norb), neleca)
    stringsb = fci.cistring.gen_strings4orblist(range(norb), nelecb)
    
    fout.write('NELACTIVE, NDETS, NORBCORE, NORBACTIVE\n')
    fout.write(' %5d %5d %5d %5d\n' % (neleca+nelecb, ndet, ncore, norb))
    fout.write('COEFFICIENT/ OCCUPIED ACTIVE SPIN ORBITALS\n')

    def str2orbidx(string):
        bstring = bin(string)
        return [i+1 for i,s in enumerate(bstring[::-1]) if s == '1']

    n = 0
    for idet, (stra, strb) in enumerate(civec[root]._strs.reshape(ndet,2,-1)):
        ka = stradic[stra[0]]
        kb = strbdic[strb[0]]
        idxa = ['%3d' % x for x in str2orbidx(stringsa[ka])]
        idxb = ['%3d' % (-x) for x in str2orbidx(stringsb[kb])]
        if (abs(civec[root][idet]) >= 1e-6):
            n = n + 1
            fout.write('%18.10E %s %s\n' % (civec[root][idet], ' '.join(idxa), ' '.join(idxb)))
    fout.write('The purged number of dets is : %d\n' % n)

def to_fci_wfn_gs_1(fout, civec, norb, nelec, root=0, ncore=0):
    if isinstance(nelec, (int, numpy.number)):
        nelecb = nelec//2
        neleca = nelec - nelecb
    else:
        neleca, nelecb = nelec
    strsa = cistring.gen_strings4orblist(range(norb), neleca)
    stradic = dict(zip(strsa,range(strsa.__len__())))
    strsb = cistring.gen_strings4orblist(range(norb), nelecb)
    strbdic = dict(zip(strsb,range(strsb.__len__())))
    na = len(stradic)
    nb = len(strbdic)
    ndet = len(civec)
    from pyscf import fci
    stringsa = fci.cistring.gen_strings4orblist(range(norb), neleca)
    stringsb = fci.cistring.gen_strings4orblist(range(norb), nelecb)
    
    fout.write('NELACTIVE, NDETS, NORBCORE, NORBACTIVE\n')
    fout.write(' %5d %5d %5d %5d\n' % (neleca+nelecb, ndet, ncore, norb))
    fout.write('COEFFICIENT/ OCCUPIED ACTIVE SPIN ORBITALS\n')

    def str2orbidx(string):
        bstring = bin(string)
        return [i+1 for i,s in enumerate(bstring[::-1]) if s == '1']

    n = 0
    for idet, (stra, strb) in enumerate(civec._strs.reshape(ndet,2,-1)):
        ka = stradic[stra[0]]
        kb = strbdic[strb[0]]
        idxa = ['%3d' % x for x in str2orbidx(stringsa[ka])]
        idxb = ['%3d' % (-x) for x in str2orbidx(stringsb[kb])]
        if (abs(civec[idet]) >= 1e-6):
            n = n + 1
            fout.write('%18.10E %s %s\n' % (civec[idet], ' '.join(idxa), ' '.join(idxb)))
    fout.write('The purged number of dets is : %d\n' % n)

def to_fci_wfn_gs(fout, civec, norb, nelec, root=0, ncore=0):
    if isinstance(nelec, (int, numpy.number)):
        nelecb = nelec//2
        neleca = nelec - nelecb
    else:
        neleca, nelecb = nelec
    strsa = cistring.gen_strings4orblist(range(norb), neleca)
    stradic = dict(zip(strsa,range(strsa.__len__())))
    strsb = cistring.gen_strings4orblist(range(norb), nelecb)
    strbdic = dict(zip(strsb,range(strsb.__len__())))
    na = len(stradic)
    nb = len(strbdic)
    ndet = len(civec[root])
    from pyscf import fci
    stringsa = fci.cistring.gen_strings4orblist(range(norb), neleca)
    stringsb = fci.cistring.gen_strings4orblist(range(norb), nelecb)
    
    fout.write('NELACTIVE, NDETS, NORBCORE, NORBACTIVE\n')
    fout.write(' %5d %5d %5d %5d\n' % (neleca+nelecb, ndet, ncore, norb))
    fout.write('COEFFICIENT/ OCCUPIED ACTIVE SPIN ORBITALS\n')

    def str2orbidx(string):
        bstring = bin(string)
        return [i+1 for i,s in enumerate(bstring[::-1]) if s == '1']

    n = 0
    for idet, (stra, strb) in enumerate(civec[root]._strs.reshape(ndet,2,-1)):
        ka = stradic[stra[0]]
        kb = strbdic[strb[0]]
        idxa = ['%3d' % x for x in str2orbidx(stringsa[ka])]
        idxb = ['%3d' % (-x) for x in str2orbidx(stringsb[kb])]
        if (abs(civec[root][idet]) >= 1e-6):
            n = n + 1
            fout.write('%18.10E %s %s\n' % (civec[root][idet], ' '.join(idxa), ' '.join(idxb)))
    fout.write('The purged number of dets is : %d\n' % n)

def make_rdm12(civec, norb, nelec):
    '''Spin orbital 1- and 2-particle reduced density matrices (aa, bb, aaaa, aabb, bbbb)
    '''
    strs = civec._strs
    ndet = len(strs)
    rdm1a = numpy.zeros(norb*norb)
    rdm1b = numpy.zeros(norb*norb)
    rdm2aa = numpy.zeros(norb*norb*norb*norb)
    rdm2ab = numpy.zeros(norb*norb*norb*norb)
    rdm2bb = numpy.zeros(norb*norb*norb*norb)

    civec = numpy.asarray(civec, order='C')
    strs = numpy.asarray(strs, order='C')
    rdm1a  = numpy.asarray(rdm1a, order='C')
    rdm1b  = numpy.asarray(rdm1b, order='C')
    rdm2aa = numpy.asarray(rdm2aa, order='C')
    rdm2ab = numpy.asarray(rdm2ab, order='C')
    rdm2bb = numpy.asarray(rdm2bb, order='C')

    # Compute 1- and 2-RDMs
    libhci.compute_rdm12s(ctypes.c_int(norb), 
                          ctypes.c_int(nelec[0]), 
                          ctypes.c_int(nelec[1]), 
                          strs.ctypes.data_as(ctypes.c_void_p), 
                          civec.ctypes.data_as(ctypes.c_void_p), 
                          ctypes.c_ulonglong(ndet), 
                          rdm1a.ctypes.data_as(ctypes.c_void_p),
                          rdm1b.ctypes.data_as(ctypes.c_void_p),
                          rdm2aa.ctypes.data_as(ctypes.c_void_p),
                          rdm2ab.ctypes.data_as(ctypes.c_void_p),
                          rdm2bb.ctypes.data_as(ctypes.c_void_p))

    rdm1a = rdm1a.reshape([norb]*2)
    rdm1b = rdm1b.reshape([norb]*2)
    rdm2aa = rdm2aa.reshape([norb]*4)
    rdm2ab = rdm2ab.reshape([norb]*4)
    rdm2bb = rdm2bb.reshape([norb]*4)

    # Sort 2-RDM into chemists' notation: <p_1 q_2|r_1 s_2> -> (p_1 r_1| q_2 s_2)
    rdm2aa = rdm2aa.transpose(0,2,1,3)
    rdm2ab = rdm2ab.transpose(0,2,1,3)
    rdm2bb = rdm2bb.transpose(0,2,1,3)

    return rdm1a+rdm1b, rdm2aa+rdm2ab+rdm2ab.transpose(2,3,0,1)+rdm2bb

