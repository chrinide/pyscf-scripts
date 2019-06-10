#!/usr/bin/env python

import numpy
from pyscf import gto, scf, lib, ao2mo
einsum = lib.einsum

mol = gto.Mole()
mol.basis = '6-31g'
mol.atom = '''
O
H 1 1.1
H 1 1.1 2 104
'''
mol.charge = 0
mol.spin = 0
mol.symmetry = 1
mol.verbose = 4
mol.build()

mf = scf.UHF(mol)
ehf = mf.scf()

orbea = mf.mo_energy[0]
orbeb = mf.mo_energy[1]
occ_a = mf.mo_occ[0]
occ_b = mf.mo_occ[1]
mo_a = mf.mo_coeff[0]
mo_b = mf.mo_coeff[1]
nmoa = mo_a.shape[1]
nmob = mo_b.shape[1]

eriaa = ao2mo.kernel(mf._eri, mo_a, compact=False).reshape([nmoa]*4)
eribb = ao2mo.kernel(mf._eri, mo_b, compact=False).reshape([nmob]*4)
eriab = ao2mo.kernel(mf._eri, (mo_a,mo_a,mo_b,mo_b), compact=False)
eriab = eriab.reshape([nmoa,nmoa,nmob,nmob])

nbf = nmoa
dalpha = numpy.zeros((nbf))
dbeta = numpy.zeros((nbf))
nalpha = mol.nelectron//2
nbeta = nalpha
aoccs = range(nalpha)
boccs = range(nbeta)
avirt = range(nalpha,nbf) 
bvirt = range(nbeta,nbf)

########  Computation of the primary energy correction  ######### 
# Simplified Size-Consistent Configuration Interaction
# This may be equivalent to second-order Epstein-Nesbet pair
# correlation theory

ec1 = 0.0
sum = 0.0
z = 1.0

#compute correction term for two alpha electrons

for a in aoccs:
    for b in range(a):
        for r in avirt:
            for s in range(nalpha,r):
                arbs = eriaa[a,r,b,s]
                asbr = eriaa[a,s,b,r]
                rraa = eriaa[r,r,a,a] - \
                       eriaa[r,a,a,r]
                rrbb = eriaa[r,r,b,b] - \
                       eriaa[r,b,b,r]
                ssaa = eriaa[s,s,a,a] - \
                       eriaa[s,a,a,s]
                ssbb = eriaa[s,s,b,b] - \
                       eriaa[s,b,b,s]
                rrss = eriaa[r,r,s,s] - \
                       eriaa[r,s,s,r]
                aabb = eriaa[a,a,b,b] - \
                       eriaa[a,b,b,a]

                eigendif = (orbea[r] + orbea[s] - orbea[a] - orbea[b])
                delcorr = (-rraa - rrbb - ssaa - ssbb + rrss + aabb) 
                delta = eigendif + delcorr*z

                eio = (arbs - asbr)

                x = -eio/delta
                if (abs(x) > 1):
                    lib.logger.info(mf,"Warning a large x value has been discovered with x = " % x)
                x = numpy.choose(x<1, (1,x))
                x = numpy.choose(x>-1, (-1,x))                   
                sum += x*x
                dalpha[a] -= x*x
                dalpha[b] -= x*x
                dalpha[r] += x*x
                dalpha[s] += x*x
                ec1 += x*eio             


#compute correction term for two beta electrons

for a in boccs:
    for b in range(a):
        for r in bvirt:
            for s in range(nbeta,r):
                arbs = eribb[a,r,b,s]
                asbr = eribb[a,s,b,r]
                rraa = eribb[r,r,a,a] - \
                       eribb[r,a,a,r]
                rrbb = eribb[r,r,b,b] - \
                       eribb[r,b,b,r]
                ssaa = eribb[s,s,a,a] - \
                       eribb[s,a,a,s]
                ssbb = eribb[s,s,b,b] - \
                       eribb[s,b,b,s]
                rrss = eribb[r,r,s,s] - \
                       eribb[r,s,s,r]
                aabb = eribb[a,a,b,b] - \
                       eribb[a,b,b,a]


                eigendif = (orbeb[r] + orbeb[s] - orbeb[a] - orbeb[b])
                delcorr = (-rraa - rrbb - ssaa - ssbb + rrss + aabb) 
                delta = eigendif + delcorr*z

                eio = (arbs - asbr)

                x = -eio/delta
                if (abs(x) > 1): 
                    lib.logger.info(mf,"Warning a large x value has been discovered with x = " % x)
                x = numpy.choose(x<1, (1,x))
                x = numpy.choose(x>-1, (-1,x))                   
                sum += x*x
                dbeta[a] -= x*x
                dbeta[b] -= x*x
                dbeta[r] += x*x
                dbeta[s] += x*x
                ec1 += x*eio

#compute correction term for one alpha and one beta electron

for a in aoccs:
    for b in boccs:
        for r in avirt:
            for s in bvirt:
                arbs = eriab[a,r,b,s]
                rraa = eriaa[r,r,a,a] - \
                       eriaa[r,a,a,r]
                rrbb = eriab[r,r,b,b]
                aass = eriab[a,a,s,s]
                ssbb = eribb[s,s,b,b] - \
                       eribb[s,b,b,s]
                rrss = eriab[r,r,s,s]
                aabb = eriab[a,a,b,b]

                eigendif = (orbea[r] + orbeb[s] - orbea[a] - orbeb[b])
                delcorr = (-rraa - rrbb - aass - ssbb + rrss + aabb)
                delta = eigendif + delcorr*z

                eio = arbs

                x = -eio/delta
                if (abs(x) > 1): 
                    lib.logger.info(mf,"Warning a large x value has been discovered with x = " % x)
                x = numpy.choose(x<1, (1,x))
                x = numpy.choose(x>-1, (-1,x))                   
                sum += x*x
                dalpha[a] -= x*x
                dbeta[b] -= x*x
                dalpha[r] += x*x
                dbeta[s] += x*x
                ec1 += x*eio

#compute the fractional occupations
for a in aoccs:
    dalpha[a] += 1.0
for b in boccs:
    dbeta[b] += 1.0
lib.logger.info(mf,"Natural ocupations")
for a in range(nbf):
    lib.logger.info(mf,"Alpha,beta %12.6f,%12.6f" % (dalpha[a], dbeta[a]))

e = ehf + ec1
lib.logger.info(mf,"The total sum of excitations is %12.6f" % sum)
lib.logger.info(mf,"The primary correlation correction is %12.6f" % ec1)
lib.logger.info(mf,"The total energy is %12.6f", e)

