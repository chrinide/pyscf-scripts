#!/usr/bin/env python

import sys
import numpy

def read_aom(file1, norb):
    f2 = open(file1, 'r')
    one_pdm = numpy.zeros((norb, norb))
    norbp = norb*(norb+1)/2
    tmp = numpy.zeros((norbp))
    i = 0
    for line in f2:
        line = line.strip()
        for number in line.split():
            tmp[i] = float(number)
            i += 1

    ij = 0
    for i in range(norb):
        for j in range(i+1):
            one_pdm[i,j] = tmp[ij]
            one_pdm[j,i] = tmp[ij]
            ij += 1

    #for i in range(norb):
    #    for j in range(norb):
    #        print i,j,one_pdm[i,j]

    f2.close()

    return one_pdm

def read1(file1, norb_active):
    f2 = open(file1, 'r')
    one_pdm = numpy.zeros((norb_active, norb_active))
    for line in f2.readlines():
        linesp = line.split()
        ind1 = int(linesp[0])
        ind2 = int(linesp[1])
        ind1 = ind1 - 1
        ind2 = ind2 - 1
        one_pdm[ind1, ind2] = float(linesp[2])

    f2.close()

    return one_pdm

def read2(file2, norb_active):
    f2 = open(file2, 'r')
    two_pdm = numpy.zeros((norb_active, norb_active, norb_active, norb_active))
    for line in f2.readlines():
        linesp = line.split()
        ind1 = int(linesp[0])
        ind2 = int(linesp[1])
        ind3 = int(linesp[2])
        ind4 = int(linesp[3])
        ind1 = ind1 - 1
        ind2 = ind2 - 1
        ind3 = ind3 - 1
        ind4 = ind4 - 1
        two_pdm[ind1, ind2, ind3, ind4] = float(linesp[4])

    f2.close()

    return two_pdm

name = 'h2o'
nmo = 5
pair = numpy.zeros((2),dtype=numpy.int)
z = numpy.zeros((2))
pair[0] = 1
pair[1] = 2
z[0] = 8.0
z[1] = 1.0
s1 = read_aom(name+'.wfn.aom'+str(pair[0]), nmo)
s2 = read_aom(name+'.wfn.aom'+str(pair[1]), nmo)

ifile = name + '.overlap'
s = read1(ifile, nmo)
ifile = name + '.kinetic'
t = read1(ifile, nmo)
ifile = name + '.nucelec'
v = read1(ifile, nmo)
ifile = name + '.rdm1'
rdm1 = read1(ifile, nmo)
ifile = name + '.rdm2_xc'
rdm2_xc = read2(ifile, nmo)
ifile = name + '.eri'
eri_mo = read2(ifile, nmo)

print('\nPopulation and charges')
print('###################################')
rho1 = numpy.einsum('ij,ji->ij',rdm1,s1)
rho2 = numpy.einsum('ij,ji->ij',rdm1,s2)
pop1 = numpy.einsum('ij->',rho1)
pop2 = numpy.einsum('ij->',rho2)
q1 = z[0]-pop1 
q2 = z[1]-pop2 
print('Population and charge of basin ', pair[0], ' : ', pop1, q1)
print('Population and charge of basin ', pair[1], ' : ', pop2, q2)

