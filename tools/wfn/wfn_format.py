#!/usr/bin/env python

from functools import reduce
from pyscf import gto
from pyscf import lib
import numpy

# types
# 1 S
# 2 PX
# 3 PY
# 4 PZ
# 5 DXX
# 6 DYY
# 7 DZZ
# 8 DXY
# 9 DXZ
# 10 DYZ
# 11 FXXX
# 12 FYYY
# 13 FZZZ
# 14 FXXY
# 15 FXXZ
# 16 FYYZ
# 17 FXYY
# 18 FXZZ
# 19 FYZZ
# 20 FXYZ
# 21 GXXXX
# 22 GYYYY
# 23 GZZZZ
# 24 GXXXY
# 25 GXXXZ
# 26 GXYYY
# 27 GYYYZ
# 28 GXZZZ
# 29 GYZZZ
# 30 GXXYY
# 31 GXXZZ
# 32 GYYZZ
# 33 GXXYZ
# 34 GXYYZ
# 35 GXYZZ
# 36 HZZZZZ
# 37 HYZZZZ
# 38 HYYZZZ
# 39 HYYYZZ
# 40 HYYYYZ
# 41 HYYYYY
# 42 HXZZZZ
# 43 HXYZZZ
# 44 HXYYZZ
# 45 HXYYYZ
# 46 HXYYYY
# 47 HXXZZZ
# 48 HXXYZZ
# 49 HXXYYZ
# 50 HXXYYY
# 51 HXXXZZ
# 52 HXXXYZ
# 53 HXXXYY
# 54 HXXXXZ
# 55 HXXXXY
# 56 HXXXXX
TYPE_MAP = [
    [1],  # S
    [2,3,4],  # P
    [5,8,9,6,10,7],  # D
    [11,14,15,17,20,18,12,16,19,13],  # F
    [21,24,25,30,33,31,26,34,35,28,22,27,32,29,23],  # G
    [56,55,54,53,52,51,50,49,48,47,46,45,44,43,42,41,40,39,38,37,36],  # H
]

def write_mo(fout, mol, mo_coeff, mo_occ, mo_energy=None):
    nmo = mo_coeff.shape[1]
    mo_cart = []
    centers = []
    types = []
    exps = []
    p0 = 0
    for ib in range(mol.nbas):
        ia = mol.bas_atom(ib)
        l = mol.bas_angular(ib)
        es = mol.bas_exp(ib)
        c = mol._libcint_ctr_coeff(ib)
        np, nc = c.shape
        nd = nc*(2*l+1)
        mosub = mo_coeff[p0:p0+nd].reshape(-1,nc,nmo)
        c2s = gto.cart2sph(l)
        mosub = numpy.einsum('yki,cy,pk->pci', mosub, c2s, c)
        mo_cart.append(mosub.transpose(1,0,2).reshape(-1,nmo))

        for t in TYPE_MAP[l]:
            types.append([t]*np)
        ncart = gto.len_cart(l)
        exps.extend([es]*ncart)
        centers.extend([ia+1]*(np*ncart))
        p0 += nd
    mo_cart = numpy.vstack(mo_cart)
    centers = numpy.hstack(centers)
    types = numpy.hstack(types)
    exps = numpy.hstack(exps)
    nprim, nmo = mo_cart.shape

    fout.write('From PySCF\n')
    fout.write('GAUSSIAN            %3d MOL ORBITALS    %3d PRIMITIVES      %3d NUCLEI\n'
            % (mo_cart.shape[1], mo_cart.shape[0], mol.natm))
    for ia in range(mol.natm):
        x, y, z = mol.atom_coord(ia)
        fout.write('  %-4s %-4d (CENTRE%3d)  %11.8f %11.8f %11.8f  CHARGE = %.1f\n'
                % (mol.atom_pure_symbol(ia), ia+1, ia+1, x, y, z,
                    mol.atom_charge(ia)))
    for i0, i1 in lib.prange(0, nprim, 20):
        fout.write('CENTRE ASSIGNMENTS  %s\n' % ''.join('%3d'%x for x in centers[i0:i1]))
    for i0, i1 in lib.prange(0, nprim, 20):
        fout.write('TYPE ASSIGNMENTS    %s\n' % ''.join('%3d'%x for x in types[i0:i1]))
    for i0, i1 in lib.prange(0, nprim, 5):
        fout.write('EXPONENTS  %s\n' % ' '.join('%13.7E'%x for x in exps[i0:i1]))

    for k in range(nmo):
        mo = mo_cart[:,k]
        if mo_energy is None:
            fout.write('MO  %-4d                  OCC NO = %12.8f ORB. ENERGY = 0.0000\n' %
                    (k+1, mo_occ[k]))
            for i0, i1 in lib.prange(0, nprim, 5):
                fout.write(' %s\n' % ' '.join('%15.8E'%x for x in mo[i0:i1]))
        else:
            fout.write('MO  %-4d                  OCC NO = %12.8f ORB. ENERGY = %12.8f\n' %
                    (k+1, mo_occ[k], mo_energy[k]))
            for i0, i1 in lib.prange(0, nprim, 5):
                fout.write(' %s\n' % ' '.join('%15.8E'%x for x in mo[i0:i1]))
    fout.write('END DATA\n')
    #if mo_energy is None:
    #    fout.write('ALDET    ENERGY =        0.0000000000   VIRIAL(-V/T)  =   0.00000000\n')
    #else :
    #    fout.write('RHF      ENERGY =        0.0000000000   VIRIAL(-V/T)  =   0.00000000\n')

def write_ci(fout, fcivec, norb, nelec, ncore=0):
    from pyscf import fci
    if isinstance(nelec, (int, numpy.number)):
        nelecb = nelec//2
        neleca = nelec - nelecb
    else:
        neleca, nelecb = nelec
    fout.write('NELACTIVE, NDETS, NORBCORE, NORBACTIVE\n')
    fout.write(' %5d %5d %5d %5d\n' % (neleca+nelecb, fcivec.size, ncore, norb))
    fout.write('COEFFICIENT/ OCCUPIED ACTIVE SPIN ORBITALS\n')

    nb = fci.cistring.num_strings(norb, nelecb)
    stringsa = fci.cistring.gen_strings4orblist(range(norb), neleca)
    stringsb = fci.cistring.gen_strings4orblist(range(norb), nelecb)
    def str2orbidx(string, ncore):
        bstring = bin(string)
        return [i+1+ncore for i,s in enumerate(bstring[::-1]) if s == '1']

    addrs = numpy.argsort(abs(fcivec.ravel()))
    n = 0
    ncore = 0
    for iaddr in reversed(addrs):
        addra, addrb = divmod(iaddr, nb)
        idxa = ['%3d' % x for x in str2orbidx(stringsa[addra], ncore)]
        idxb = ['%3d' % (-x) for x in str2orbidx(stringsb[addrb], ncore)]
        if ( abs(fcivec[addra,addrb]) >= 1e-6 ):
            n = n + 1
            fout.write('%18.10E %s %s\n' % (fcivec[addra,addrb], ' '.join(idxa), ' '.join(idxb)))
    fout.write('The purged number of dets is : %d\n' % n)

def write_ci_hubbard(fout, fcivec, norb, nelec, ncore=0):
    from pyscf import fci
    if isinstance(nelec, (int, numpy.number)):
        nelecb = nelec//2
        neleca = nelec - nelecb
    else:
        neleca, nelecb = nelec
    fout.write('NELACTIVE, NDETS, NORBCORE, NORBACTIVE\n')
    fout.write(' %5d %5d %5d %5d\n' % (neleca+nelecb, fcivec.size, ncore, norb))
    fout.write('COEFFICIENT/ OCCUPIED ACTIVE SPIN ORBITALS\n')

    nb = fci.cistring.num_strings(norb, nelecb)
    stringsa = fci.cistring.gen_strings4orblist(range(norb), neleca)
    stringsb = fci.cistring.gen_strings4orblist(range(norb), nelecb)
    def str2orbidx(string, ncore):
        bstring = bin(string)
        return [i+1+ncore for i,s in enumerate(bstring[::-1]) if s == '1']

    addrs = numpy.argsort(abs(fcivec.ravel()))
    ncore = 0
    for iaddr in reversed(addrs):
        addra, addrb = divmod(iaddr, nb)
        idxa = ['%3d' % x for x in str2orbidx(stringsa[addra], ncore)]
        idxb = ['%3d' % (-x) for x in str2orbidx(stringsb[addrb], ncore)]
        fout.write('%18.10E %s %s\n' % (fcivec[addra,addrb], ' '.join(idxa), ' '.join(idxb)))

def write_coeff(fout, mol, mo_coeff):
    fout.write('ALDET\n')
    nmo = mo_coeff.shape[1]
    mo_cart = []
    centers = []
    types = []
    exps = []
    p0 = 0
    for ib in range(mol.nbas):
        ia = mol.bas_atom(ib)
        l = mol.bas_angular(ib)
        es = mol.bas_exp(ib)
        c = mol._libcint_ctr_coeff(ib)
        #c = mol.bas_ctr_coeff(ib)
        np, nc = c.shape
        nd = nc*(2*l+1)
        mosub = mo_coeff[p0:p0+nd].reshape(-1,nc,nmo)
        c2s = gto.cart2sph(l)
        mosub = numpy.einsum('yki,cy,pk->pci', mosub, c2s, c)
        mo_cart.append(mosub.transpose(1,0,2).reshape(-1,nmo))

        for t in TYPE_MAP[l]:
            types.append([t]*np)
        ncart = gto.len_cart(l)
        exps.extend([es]*ncart)
        centers.extend([ia+1]*(np*ncart))
        p0 += nd
    mo_cart = numpy.vstack(mo_cart)
    centers = numpy.hstack(centers)
    types = numpy.hstack(types)
    exps = numpy.hstack(exps)
    nprim, nmo = mo_cart.shape

    fout.write('From PySCF\n')
    for k in range(nmo):
        mo = mo_cart[:,k]
        fout.write('CANMO %d\n' % (k+1))
        for i0, i1 in lib.prange(0, nprim, 5):
            fout.write(' %s\n' % ' '.join('%15.8E'%x for x in mo[i0:i1]))
    fout.write('END DATA\n')

def write_hci(fout, civec, norb, nelec, root=0, ncore=0):
    from pyscf.fci import cistring
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

