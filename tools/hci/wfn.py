#!/usr/bin/env python

import numpy
from pyscf.fci import cistring

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

