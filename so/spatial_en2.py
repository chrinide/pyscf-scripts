#!/usr/bin/env python

import numpy
from pyscf import gto, scf, lib, ao2mo
einsum = lib.einsum

mol = gto.Mole()
mol.verbose = 4
mol.atom = '''
O      0.000000      0.000000      0.118351
H      0.000000      0.761187     -0.469725
H      0.000000     -0.761187     -0.469725
'''
mol.basis = 'cc-pvtz'
mol.symmetry = 1
mol.spin = 0
mol.charge = 0
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
Yalpha = numpy.zeros((nbf))
Ybeta = numpy.zeros((nbf))
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

Ec1 = 0.
sum = 0.
z = 1.

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

                Eio = (arbs - asbr)

                x = -Eio/delta
                if abs(x) > 1:
                    lib.logger.info(mf,"Warning a large x value has been discovered with x = " % x)
                x = numpy.choose(x < 1, (1,x))
                x = numpy.choose(x > -1, (-1,x))                   
                sum += x*x
                Yalpha[a] -= x*x
                Yalpha[b] -= x*x
                Yalpha[r] += x*x
                Yalpha[s] += x*x
                Ec1 += x*Eio             


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

                Eio = (arbs - asbr)

                x = -Eio/delta
                if abs(x) > 1: 
                    lib.logger.info(mf,"Warning a large x value has been discovered with x = " % x)
                x = numpy.choose(x < 1, (1,x))
                x = numpy.choose(x > -1, (-1,x))                   
                sum += x*x
                Ybeta[a] -= x*x
                Ybeta[b] -= x*x
                Ybeta[r] += x*x
                Ybeta[s] += x*x
                Ec1 += x*Eio

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

                Eio = arbs

                x = -Eio/delta
                if abs(x) > 1: 
                    lib.logger.info(mf,"Warning a large x value has been discovered with x = " % x)
                x = numpy.choose(x < 1, (1,x))
                x = numpy.choose(x > -1, (-1,x))                   
                sum += x*x
                Yalpha[a] -= x*x
                Ybeta[b] -= x*x
                Yalpha[r] += x*x
                Ybeta[s] += x*x
                Ec1 += x*Eio

#compute the fractional occupations
for a in aoccs:
    Yalpha[a] = 1 + Yalpha[a]
for b in boccs:
    Ybeta[b] = 1 + Ybeta[b]
for a in range(nbf):
    print "For alpha = ",a,"the fractional occupation is ",Yalpha[a]
    print "For beta  = ",a,"the fractional occupation is ",Ybeta[a]

#print the energy and its corrections
E = ehf + Ec1
print "The total sum of excitations is ",sum
print "The primary correlation correction is ",Ec1
print "The total energy is ", E
