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

def print_dets(self):
    logger.info(self, '*** Printing list of determinants')
    logger.info(self, 'Number of determinants: %s', self.ndets)
    for i in range(self.ndets):
        logger.info(self,'Det %d alpha %s, beta %s' % \
        (i,bin(self.strs[i,0]),bin(self.strs[i,1])))
    return self

def gen_cis(self):
    neleca = self.nelec[0]
    nocca = self.nelec[0]
    nvira = self.norb - self.nelec[0]
    nelecb = self.nelec[1]
    noccb = self.nelec[1]
    nvirb = self.norb - self.nelec[1]
    hf_stra = int('1'*neleca, 2)
    hf_strb = int('1'*nelecb, 2)
    ndetsa = neleca*nvira
    ndetsb = nelecb*nvirb

    self.ndets = ndetsa + ndetsb + 1
    logger.info(self, 'Number of determinants: %s', self.ndets)
    self.strs = numpy.empty((self.ndets,2), dtype=numpy.uint64)
    self.strs[0,0] = hf_stra
    self.strs[0,1] = hf_strb
    k = 1
    alphao, alphau = str2orblst(self.strs[0,0], self.norb)
    for i in alphao:
        for j in alphau:
            stra = rmorb(self.strs[0,0], i)
            stra = addorb(stra, j)
            self.strs[k,0] = stra
            self.strs[k,1] = hf_strb
            k += 1
    betao, betau = str2orblst(self.strs[0,1], self.norb)
    for i in betao:
        for j in betau:
            strb = rmorb(self.strs[0,1], i)
            strb = addorb(strb, j)
            self.strs[k,0] = hf_stra
            self.strs[k,1] = strb
            k += 1

    return self

# TODO: recheck the dets still experimental
def gen_cisd(self):
    neleca = self.nelec[0]
    nocca = self.nelec[0]
    nvira = self.norb - self.nelec[0]
    nelecb = self.nelec[1]
    noccb = self.nelec[1]
    nvirb = self.norb - self.nelec[1]
    hf_stra = int('1'*neleca, 2)
    hf_strb = int('1'*nelecb, 2)

    ndet_s = (nocca*nvira)+(noccb+nvirb)
    ndet_d = num_strings(nocca,2) * num_strings(nvira, 2) * \
             num_strings(noccb,2) * num_strings(nvirb, 2) + nocca*noccb * nvira*nvirb
    self.ndets = ndet_s + ndet_d

    logger.info(self, 'Number of determinants: %s', self.ndets)
    self.strs = numpy.empty((self.ndets,2), dtype=numpy.uint64)
    self.strs[0,0] = hf_stra
    self.strs[0,1] = hf_strb

    kk = 1
    alphao, alphau = str2orblst(self.strs[0,0], self.norb)
    betao, betau = str2orblst(self.strs[0,1], self.norb)

    for i in alphao:
        for j in alphau:
            stra = rmorb(self.strs[0,0], i)
            stra = addorb(stra, j)
            self.strs[kk,0] = stra
            self.strs[kk,1] = hf_strb
            kk += 1

    for i in betao:
        for j in betau:
            strb = rmorb(self.strs[0,1], i)
            strb = addorb(strb, j)
            self.strs[kk,0] = hf_stra
            self.strs[kk,1] = strb
            kk += 1

    for i in alphao:
        for j in alphau:
            for k in betao:
                for l in betau:
                    stra = rmorb(self.strs[0,0], i)
                    stra = addorb(stra, j)
                    strb = rmorb(self.strs[0,1], k)
                    strb = addorb(strb, l)
                    self.strs[kk,0] = stra
                    self.strs[kk,1] = strb
                    kk += 1

    for i1, i2 in combinations(alphao, 2):
        for j1, j2 in combinations(alphau, 2):
            stra = rmorb(self.strs[0,0], i1)
            stra = addorb(stra, i2)
            stra = rmorb(stra, j1)
            stra = addorb(stra, j2)
            self.strs[kk,0] = stra
            self.strs[kk,1] = hf_strb 
            kk += 1

    for k1, k2 in combinations(betao, 2):
        for l1, l2 in combinations(betau, 2):
            strb = rmorb(self.strs[0,1], k1)
            strb = addorb(strb, k2)
            strb = rmorb(strb, l1)
            strb = addorb(strb, l2)
            self.strs[kk,0] = hf_strb
            self.strs[kk,1] = strb 
            kk += 1

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
                    #print "a", ip,jp,i,a,fai,sign
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
                    #print "b", ip,jp,i,a,fai,sign
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
                    #print "aa", i,a,v,sign
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
                    #print "bb", i,a,v,sign
                    h[ip,jp] = sign * v
                    h[jp,ip] = h[ip,jp]
# alpha,beta ->alpha,beta
                else:
                    i,a = desa[0], crea[0]
                    j,b = desb[0], creb[0]
                    v = self.h2e[a,i,b,j]
                    sign = cre_des_sign(a, i, stria)
                    sign*= cre_des_sign(b, j, strib)
                    #print "ab", ip,jp,i,a,v,sign
                    h[ip,jp] = 1*sign * v
                    h[jp,ip] = h[ip,jp]
    #print self.h1e
    #h[abs(h) < 1e-12] = 0
    #print h                    
    return h
    
class det(lib.StreamObject):
    def __init__(self, mf, nelec, norb):
        self.stdout = mf.stdout
        self.verbose = mf.verbose
        self.mf = mf
        self.mol = mf.mol
        self.nelec = nelec
        self.norb = norb
        self.model = 'cis'
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
        if (self.model == 'cis'):
            gen_cis(self)
        elif (self.model == 'cisd'):
            gen_cisd(self)
        else:
            raise RuntimeError('''CIS only available at this moment''')
        eri_size = (self.norb**4)*8e-9
        ham_size = (self.ndets+self.ndets**2)*8e-9
        tot_size = eri_size + 2.0*ham_size
        logger.info(self, 'Estimated memoryi GB: %s', tot_size)
        print_dets(self)
        return self

    def build_h(self):
        h = numpy.zeros((self.ndets, self.ndets), dtype=numpy.complex128)
        h = make_hdiag(self, h)
        h= make_hoffdiag(self, h)
        e,c = numpy.linalg.eigh(h)
        print e[0]+self.e_core
        #print c[:,0]
        #print e[1]+self.e_core
        #print c[:,1]
        #print e+self.e_core
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
        norb = self.norb
        self.h2e = ao2mo.kernel(self.mf.mol, coeff[:, ci_idx], intor='int2e_spinor').reshape(norb,norb,norb,norb)
        #self.h2e = ao2mo.restore(1, self.h2e, self.norb)
        self.gen_strs()
        self.build_h()
        return self

if __name__ == '__main__':
    from pyscf import gto, scf, mcscf, tdscf, x2c
    mol = gto.Mole()
    mol.basis = 'cc-pvdz'
    mol.atom = '''
    H 0.0000  0.0000  0.0000
    H 0.0000  0.0000  0.7500
    '''
    mol.verbose = 4
    mol.spin = 0
    mol.charge = 0
    mol.symmetry = 1
    mol.build()

    mf = x2c.RHF(mol)
    mf.kernel()

    #td = tdscf.TDA(mf)
    #ex = td.kernel()[0]
    #print ex

    norb = 10
    nelec = (1,1)
    dets = det(mf,nelec,norb)
    dets.model = 'cisd'
    dets.kernel()

    #mc = mcscf.CASCI(mf,2,2)
    #mc.fcisolver.nroots = 2
    #mc.kernel()
