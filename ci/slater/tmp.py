#!/usr/bin/env python
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

def addorb(string, idx):
    string0 = string
    string0 |= (1<<idx)
    return string0

def rmorb(string, idx):
    string0 = string
    string0 &= ~(1<<idx)
    return string0


def tn_strs(self,norb,nelec,n):
    if nelec < n or norb-nelec < n:
        return numpy.zeros(0, dtype=numpy.int64)
    occs_allow = numpy.asarray(make_strings(self,range(nelec),n)[::-1])
    virs_allow = numpy.asarray(make_strings(self,range(nelec,norb),n))
    hf_str = int('1'*nelec, 2)
    tns = (hf_str | virs_allow.reshape(-1,1)) ^ occs_allow
    return tns.ravel()
