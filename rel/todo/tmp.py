#!/usr/bin/env python

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
