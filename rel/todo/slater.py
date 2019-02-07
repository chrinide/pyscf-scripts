#!/usr/bin/env python

import numpy
import math
from itertools import combinations

from pyscf import scf, lib, ao2mo
from pyscf.lib import logger

def find1(s):
    return [i for i,x in enumerate(bin(s)[2:][::-1]) if x is '1']

def str2orblst(string, norb):
    occ = []
    vir = []
    occ.extend([x for x in find1(string)])
    for i in range(norb): 
        if not (string & numpy.uint64(1<<i)):
            vir.append(i)
    return occ, vir

def orblst2str(lst):
    string = numpy.uint64(0)
    for idx in lst:
        string ^= numpy.uint64(1<<idx)
    return string

def addorb(string, idx):
    string0 = string
    string0 |= numpy.uint64(1<<idx)
    return string0

def rmorb(string, idx):
    string0 = string
    string0 &= ~numpy.uint64((1<<idx))
    return string0

def str_diff(string0, string1):
    des_string0 = []
    cre_string0 = []
    df = string0 ^ string1
    des_string0.extend([x for x in find1(df & string0)])
    cre_string0.extend([x for x in find1(df & string1)])
    return des_string0, cre_string0    

def num_strings(n, m):
    if m < 0 or m > n:
        return 0
    else:
        return math.factorial(n) // (math.factorial(n-m)*math.factorial(m))

def cre_des_sign(p, q, string0):
    if p == q:
        return 1
    else:
        if (string0 & numpy.uint64(1<<p)) or (not (string0 & numpy.uint64(1<<q))):
            return 0
        elif p > q:
            mask = numpy.uint64(1 << p) - numpy.uint64(1 << (q+1))
        else:
            mask = numpy.uint64(1 << q) - numpy.uint64(1 << (p+1))
        return (-1) ** bin(string0 & mask).count('1')

def print_dets(self,strs):
    ndets = strs.shape[0]
    logger.info(self, '*** Printing list of determinants')
    logger.info(self, 'Number of determinants: %s', ndets)
    for i in range(ndets):
        logger.info(self,'Det %d %s' % (i,bin(strs[i])))
    return self

def gen_cisd(self):
    nelec = self.mol.nelectron
    nocc = self.mol.nelectron
    norb = self.mo_coeff.shape[1]
    nvir = norb - self.mol.nelectron
    hf_str = int('1'*nelec, 2)

    ndet_s = nocc*nvir
    ndet_d = num_strings(nocc,2) * num_strings(nvir,2)
    ndets = ndet_s + ndet_d + 1

    logger.info(self, 'Number of determinants: %s', ndets)
    strs = numpy.empty((ndets), dtype=numpy.uint64)
    strs[0] = hf_str

    kk = 1
    occ, vir = str2orblst(strs[0], norb)

    for i in occ:
        for j in vir:
            stra = rmorb(strs[0], i)
            stra = addorb(stra, j)
            strs[kk] = stra
            kk += 1
    
    for i1, i2 in combinations(occ, 2):
        for j1, j2 in combinations(vir, 2):
            stra = rmorb(strs[0], i1)
            stra = addorb(stra, i2)
            stra = rmorb(stra, j1)
            stra = addorb(stra, j2)
            strs[kk] = stra
            kk += 1

    return strs

def make_hdiag(self,h1e,h2e,h,strs):
    ndets = strs.shape[0]
    norb = h1e.shape[0]
    diagj = numpy.einsum('iijj->ij', h2e)
    diagk = numpy.einsum('ijji->ij', h2e)
    for i in range(ndets):
        stra = strs[i]
        occs = str2orblst(stra, norb)[0]
        e1 = h1e[occs,occs].sum()
        e2 = diagj[occs][:,occs].sum() \
           - diagk[occs][:,occs].sum()
        h[i,i] = e1 + e2*0.5
    return h

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
                #v = h2e[a,i,b,j] - h2e[a,j,b,i]
                #sign = cre_des_sign(b, j, stri)
                #sign*= cre_des_sign(a, i, stri)
                h[ip,jp] = sign * v
                h[jp,ip] = h[ip,jp].conj()
            else:
                continue
    return h

if __name__ == '__main__':
    from pyscf import gto, scf, x2c, ao2mo
    from pyscf.tools.dump_mat import dump_tri
    mol = gto.Mole()
    mol.basis = 'sto-6g'
    mol.atom = '''
    H 0.0000  0.0000  0.0000
    H 0.0000  0.0000  0.7500
    '''
    mol.verbose = 4
    mol.spin = 0
    mol.charge = 0
    mol.symmetry = 0
    mol.build()

    mf = x2c.UHF(mol)
    mf.kernel()

    e_core = mol.energy_nuc() 
    nao, nmo = mf.mo_coeff.shape
    eri_mo = ao2mo.kernel(mol, mf.mo_coeff[:,:nmo], \
    compact=False, intor='int2e_spinor')
    eri_mo = eri_mo.reshape(nmo,nmo,nmo,nmo)
    h1e = reduce(numpy.dot, (mf.mo_coeff[:,:nmo].conj().T, \
    mf.get_hcore(), mf.mo_coeff[:,:nmo]))

    strs = gen_cisd(mf)
    ndets = strs.shape[0]
    h = numpy.zeros((ndets,ndets), dtype=numpy.complex128)
    h = make_hdiag(mf,h1e,eri_mo,h,strs) 
    h = make_hoffdiag(mf,h1e,eri_mo,h,strs)
    dump_tri(mf.stdout,h,ncol=15,digits=4)

    e,c = numpy.linalg.eigh(h)
    e += e_core
    logger.info(mf, 'Ground state energy %s', e[0])
    logger.info(mf, 'Ground state civec %s', c[:,0])
    norm = numpy.einsum('i,i->',c[:,0].conj(),c[:,0])
    logger.info(mf, 'Norm of ground state civec %s', norm)

