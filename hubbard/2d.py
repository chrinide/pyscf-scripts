#!/usr/bin/env python

import numpy, sys
from pyscf import gto, scf, mcscf, ao2mo
from math import *

def hop(tparam, nsites, pbc):
    result = numpy.zeros((nsites,nsites))
    #for i in xrange(nsites - 1):
    #    result[i, i + 1] = tparam
    #    result[i + 1, i] = tparam
    #for i in xrange(nsites - 2):
    #    result[i, i + 2] = tparam/2.0
    #    result[i + 2, i] = tparam/2.0
    result[:,:] = tparam
    numpy.fill_diagonal(result, 0.0)
    if pbc == True:
        result[nsites - 1, 0] = tparam
        result[0, nsites - 1] = tparam
    #   result[nsites - 2, 0] = tparam/2.0
    #   result[0, nsites - 2] = tparam/2.0
    return result

def hop2d(nx, ny):
    n = nx * ny
    mat = numpy.zeros([n, n])

    for site1 in range(n):
        (x1, y1) = matrixindices(site1, nx, ny)
	nbrs = nearestneighbors(site1, nx, ny)
    	for site2 in range(site1):
    	    (x2, y2) = matrixindices(site2, nx, ny)
    	    if (x2, y2) in nbrs:
    	        mat[site1, site2] = -t
    	        mat[site2, site1] = -t
    return mat

def matrixindices(site, nx, ny):
    x = site % nx
    y = int(floor(float(site)/ny))
    return x, y

def nearestneighbors(site, nx, ny):
    (x,y) = matrixindices(site, nx, ny)
    left = ((x - 1)%nx, y)
    right = ((x + 1)%nx, y)
    down = (x, (y + 1)%ny)
    up = (x, (y - 1)%ny)
    nbrs = [left, right, down, up]
    return nbrs

t = 2
u = 2
nx = 2
ny = 2
n = nx * ny
h1 = hop2d(nx,ny)
print h1
filling = 1
s = numpy.eye(n)

t1 =  0.000001
t2 = -8.0
h1[0,0] = 0.0
h1[0,1] = t1
h1[0,2] = t1
h1[0,3] = t2

h1[1,0] = t1
h1[1,1] = 0.0
h1[1,2] = t2
h1[1,3] = t1

h1[2,0] = t1
h1[2,1] = t2
h1[2,2] = 0.0
h1[2,3] = t1

h1[3,0] = t2
h1[3,1] = t1
h1[3,2] = t1
h1[3,3] = 0.0

#h1 = numpy.zeros((n,n))
#for i in range(n-1):
#    h1[i,i+1] = h1[i+1,i] = t
#h1[n-1,0] = h1[0,n-1] = t # PBC

eri = numpy.zeros((n,n,n,n))
for i in range(n):
    eri[i,i,i,i] = u

mol = gto.Mole()
mol.nelectron = 2#n*filling
mol.verbose = 4
mol.spin = 0
mol.symmetry = 0
mol.charge = 0
mol.incore_anyway = True
mol.build()

mf = scf.RHF(mol)
mf.conv_tol = 1e-8
mf.level_shift = 0.1
mf.max_cycle = 150
mf.diis = True
mf.diis_space = 12
mf.get_hcore = lambda *args: h1
mf.get_ovlp = lambda *args: s
mf._eri = ao2mo.restore(8, eri, n)
mf.kernel()
dm = mf.make_rdm1()
#dm = dm[0] + dm[1]

print h1
print " "
print "#######################################"
print "Population"
print "#######################################"
pop = numpy.einsum('ij,ji->i', dm, s)
for ia in range(n):
    symb1 = 'H'
    print ia+1, symb1, pop[ia]

print " "
print "#######################################"
print "Delocalization Indexes"
print "#######################################"
pairs2 = numpy.einsum('ij,kl,li,kj->ik',dm,dm,s,s)*0.25 # XC
for ia in range(n):
    symb1 = 'H'
    for ib in range(ia+1):
        symb2 = 'H'
        if (ia == ib): 
            factor = 1.0
        if (ia != ib): 
            factor = 2.0
        print ia+1, ib+1, symb1, symb2, 2*factor*pairs2[ia,ib]

mc = mcscf.CASCI(mf, n, mol.nelectron)
mc.kernel()

rdm1, rdm2 = mc.fcisolver.make_rdm12(mc.ci, n, mol.nelectron)
rdm2 = rdm2 - numpy.einsum('ij,kl->ijkl',rdm1,rdm1) 

rdm1 = reduce(numpy.dot, (mc.mo_coeff, rdm1, mc.mo_coeff.T))

rdm2 = numpy.dot(mc.mo_coeff, rdm2.reshape(n,-1))
rdm2 = numpy.dot(rdm2.reshape(-1,n), mc.mo_coeff.T)
rdm2 = rdm2.reshape(n,n,n,n).transpose(2,3,0,1)
rdm2 = numpy.dot(mc.mo_coeff, rdm2.reshape(n,-1))
rdm2 = numpy.dot(rdm2.reshape(-1,n), mc.mo_coeff.T)
rdm2 = rdm2.reshape(n,n,n,n)
rdm2 = -rdm2

print " "
print "#######################################"
print "Population"
print "#######################################"
pop = numpy.einsum('ij,ji->i', rdm1, s)
for ia in range(n):
    symb1 = 'H'
    print ia+1, symb1, pop[ia]

pairs2 = numpy.einsum('ijkl,ij,kl->ik',rdm2,s,s)*0.5 # XC
print " "
print "#######################################"
print "Delocalization Indexes"
print "#######################################"
for ia in range(n):
    symb1 = 'H'
    for ib in range(ia+1):
        symb2 = 'H'
        if (ia == ib): 
            factor = 1.0
        if (ia != ib): 
            factor = 2.0
        print ia+1, ib+1, symb1, symb2, 2*factor*pairs2[ia,ib]

