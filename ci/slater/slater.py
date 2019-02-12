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
        if not (string & (1<<i)):
            vir.append(i)
    return occ, vir

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
        if (string0 & (1<<p)) or (not (string0 & (1<<q))):
            return 0
        elif p > q:
            mask = (1 << p) - (1 << (q+1))
        else:
            mask = (1 << q) - (1 << (p+1))
        return (-1) ** bin(string0 & mask).count('1')

def print_dets(self):
    logger.info(self, '*** Printing list of determinants')
    logger.info(self, 'Number of determinants: %s', self.ndets)
    for i in range(self.ndets):
        logger.info(self,'Det %d alpha %s, beta %s' % \
        (i,bin(self.strs[i,0]),bin(self.strs[i,1])))
    return self

def make_hdiag(self,h):
    diagj = numpy.einsum('iijj->ij', self.h2e)
    diagk = numpy.einsum('ijji->ij', self.h2e)
    for i in range(self.ndets):
        stra = self.strs[i,0]
        strb = self.strs[i,1]
        aocc = str2orblst(stra, self.norb)[0]
        bocc = str2orblst(strb, self.norb)[0]
        e1 = self.h1e[aocc,aocc].sum() + self.h1e[bocc,bocc].sum()
        e2 = diagj[aocc][:,aocc].sum() + diagj[aocc][:,bocc].sum() \
           + diagj[bocc][:,aocc].sum() + diagj[bocc][:,bocc].sum() \
           - diagk[aocc][:,aocc].sum() - diagk[bocc][:,bocc].sum()
        h[i,i] = e1 + e2*0.5
    return h

def make_hoffdiag(self,h):
    for ip in range(self.ndets):
        for jp in range(ip):
            stria, strib = self.strs[ip,0], self.strs[ip,1]
            strja, strjb = self.strs[jp,0], self.strs[jp,1]
            desa, crea = str_diff(stria, strja)
            if len(desa) > 2:
                continue
            desb, creb = str_diff(strib, strjb)
            if len(desb) + len(desa) > 2:
                continue
            if len(desa) + len(desb) == 1:
# alpha->alpha
                if len(desb) == 0:
                    i,a = desa[0], crea[0]
                    occsa = str2orblst(stria, self.norb)[0]
                    occsb = str2orblst(strib, self.norb)[0]
                    fai = self.h1e[a,i]
                    for k in occsa:
                        fai += self.h2e[k,k,a,i] - self.h2e[k,i,a,k]
                    for k in occsb:
                        fai += self.h2e[k,k,a,i]
                    sign = cre_des_sign(a, i, stria)
                    h[ip,jp] = sign * fai
                    h[jp,ip] = h[ip,jp]
# beta ->beta
                elif len(desa) == 0:
                    i,a = desb[0], creb[0]
                    occsa = str2orblst(stria, self.norb)[0]
                    occsb = str2orblst(strib, self.norb)[0]
                    fai = self.h1e[a,i]
                    for k in occsb:
                        fai += self.h2e[k,k,a,i] - self.h2e[k,i,a,k]
                    for k in occsa:
                        fai += self.h2e[k,k,a,i]
                    sign = cre_des_sign(a, i, strib)
                    h[ip,jp] = sign * fai
                    h[jp,ip] = h[ip,jp]
            else:
# alpha,alpha->alpha,alpha
                if len(desb) == 0:
                    i,j = desa
                    a,b = crea
# 6 conditions for i,j,a,b
# --++, ++--, -+-+, +-+-, -++-, +--+ 
                    if a > j or i > b:
# condition --++, ++--
                        v = self.h2e[a,j,b,i] - self.h2e[a,i,b,j]
                        sign = cre_des_sign(b, i, stria)
                        sign*= cre_des_sign(a, j, stria)
                    else:
# condition -+-+, +-+-, -++-, +--+ 
                        v = self.h2e[a,i,b,j] - self.h2e[a,j,b,i]
                        sign = cre_des_sign(b, j, stria)
                        sign*= cre_des_sign(a, i, stria)
                    h[ip,jp] = sign * v
                    h[jp,ip] = h[ip,jp]
# beta ,beta ->beta ,beta
                elif len(desa) == 0:
                    i,j = desb
                    a,b = creb
                    if a > j or i > b:
                        v = self.h2e[a,j,b,i] - self.h2e[a,i,b,j]
                        sign = cre_des_sign(b, i, strib)
                        sign*= cre_des_sign(a, j, strib)
                    else:
                        v = self.h2e[a,i,b,j] - self.h2e[a,j,b,i]
                        sign = cre_des_sign(b, j, strib)
                        sign*= cre_des_sign(a, i, strib)
                    h[ip,jp] = sign * v
                    h[jp,ip] = h[ip,jp]
# alpha,beta ->alpha,beta
                else:
                    i,a = desa[0], crea[0]
                    j,b = desb[0], creb[0]
                    v = self.h2e[a,i,b,j]
                    sign = cre_des_sign(a, i, stria)
                    sign*= cre_des_sign(b, j, strib)
                    h[ip,jp] = 1*sign * v
                    h[jp,ip] = h[ip,jp]
    return h

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
    
class det(lib.StreamObject):
    def __init__(self, mf, nelec, norb):
        self.stdout = mf.stdout
        self.verbose = mf.verbose
        self.mf = mf
        self.mol = mf.mol
        self.nelec = nelec
        self.norb = norb
        self.model = 'fci'
##################################################
# don't modify the following attributes, they are not input options
        self.ncore = None
        self.e_core = None
        self.h1e = None
        self.h2e = None
        self.ndets = None
        self.strs = None
        self._keys = set(self.__dict__.keys())

    def dump_flags(self, verbose=None):
        if self.verbose >= logger.WARN:
            self.check_sanity()
        logger.info(self, '\n *** A simple CI module')
        logger.info(self, 'CI model: %s', self.model)
        logger.info(self, 'Number of active electrons: %s', self.nelec)
        logger.info(self, 'Active orbitals: %s', self.norb)
        logger.info(self, 'Number of core orbitals: %s', self.ncore)
        logger.info(self, 'Core energy: %s', self.e_core)
        return self

    def gen_strs(self):
        if (self.model == 'fci'):
            orb_list = self.ncore+numpy.arange(self.norb)
            strsa = make_strings(self,orb_list,self.nelec[0]) 
            strsb = make_strings(self,orb_list,self.nelec[0]) 
            ndeta = len(strsa)
            ndetb = len(strsa)
            self.ndets = ndeta*ndetb
            self.strs = numpy.zeros((self.ndets,2), dtype=numpy.int64)
            kk = 0
            for i in range(ndeta):
                for j in range(ndetb):
                    self.strs[kk,0] = strsa[i]
                    self.strs[kk,1] = strsb[j]
                    kk += 1
        else:
            raise RuntimeError('''CIS only available at this moment''')
        eri_size = (self.norb**4)*8e-9
        ham_size = (self.ndets+self.ndets**2)*8e-9
        tot_size = eri_size + 2.0*ham_size
        logger.info(self, 'Estimated memoryi GB: %s', tot_size)
        print_dets(self)
        return self

    def build_h(self):
        h = numpy.zeros((self.ndets, self.ndets))
        h = make_hdiag(self, h)
        h = make_hoffdiag(self, h)
        e,c = numpy.linalg.eigh(h)
        print h
        print e[0]+self.e_core
        return self

    def kernel(self):
        if (self.norb > 64):
            raise RuntimeError('''Only support up to 64 orbitals''')
        self.ncore = self.mol.nelectron - self.nelec[0] - self.nelec[1]
        self.ncore = self.ncore//2
        e_core = self.mol.energy_nuc()
        ci_idx = self.ncore+numpy.arange(self.norb)
        coeff = self.mf.mo_coeff
        hcore = self.mf.get_hcore()
        corevhf = 0.0
        if (self.ncore != 0):
            core_idx = numpy.arange(self.ncore)
            core_dm = numpy.dot(coeff[:, core_idx], coeff[:, core_idx].T)*2.0
            e_core += numpy.einsum('ij,ji', core_dm, hcore)
            corevhf = self.mf.get_veff(mol, core_dm)
            e_core += numpy.einsum('ij,ji', core_dm, corevhf)*0.5
        self.e_core = e_core
        self.dump_flags()
        self.h1e = reduce(numpy.dot, (coeff[:, ci_idx].T, hcore + corevhf, coeff[:, ci_idx]))
        self.h2e = ao2mo.full(self.mf._eri, coeff[:, ci_idx])
        self.h2e = ao2mo.restore(1, self.h2e, self.norb)
        self.gen_strs()
        self.build_h()
        return self

if __name__ == '__main__':
    from pyscf import gto, scf, mcscf, tdscf
    mol = gto.Mole()
    mol.basis = 'sto-6g'
    mol.atom = '''
    H 0.0000  0.0000  0.0000
    H 0.0000  0.0000  9.7500
    '''
    mol.verbose = 4
    mol.spin = 0
    mol.charge = 0
    mol.symmetry = 1
    mol.build()

    mf = scf.RHF(mol).x2c()
    mf.kernel()

    norb = 2
    nelec = (1,1)
    dets = det(mf,nelec,norb)
    dets.model = 'fci'
    dets.kernel()

    mc = mcscf.CASCI(mf,2,2)
    mc.kernel()

