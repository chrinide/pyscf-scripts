#!/usr/bin/env python

import numpy
from pyscf import gto, scf, lib
from pyscf.tools import molden

name = 'h2o'

mol = gto.Mole()
mol.basis = 'cc-pvdz'
mol.atom = '''
O
H 1 1.1
H 1 1.1 2 104
'''
mol.charge = 0
mol.spin = 1
mol.charge = 1
mol.symmetry = 1
mol.verbose = 4
mol.build()

mf = scf.UHF(mol)
mf.kernel()

coeffa = mf.mo_coeff[0][:,mf.mo_occ[0]>0]
energya = mf.mo_energy[0][mf.mo_occ[0]>0]
occa = mf.mo_occ[0][mf.mo_occ[0]>0]

coeffb = mf.mo_coeff[1][:,mf.mo_occ[1]>0]
energyb = mf.mo_energy[1][mf.mo_occ[1]>0]
occb = mf.mo_occ[1][mf.mo_occ[1]>0]

occ = numpy.hstack([occa,occb])
coeff = numpy.hstack([coeffa,coeffb])
with open(name+'.mol', 'w') as f2:
    molden.header(mol, f2)
    molden.orbital_coeff(mol, f2, coeff, occ=occ)

norb = mf.mo_energy[0].size
nmoa = energya.size
nmob = energyb.size

rdm1a = numpy.zeros((nmoa,nmoa))
rdm1b = numpy.zeros((nmob,nmob))

rdm2a = numpy.zeros((nmoa,nmoa,nmoa,nmoa))
rdm2b = numpy.zeros((nmob,nmob,nmob,nmob))
rdm2ab = numpy.zeros((nmoa,nmoa,nmob,nmob))
rdm2ba = numpy.zeros((nmob,nmob,nmoa,nmoa))

for i in range(energya.size):
    rdm1a[i,i] = 1.0

for i in range(energyb.size):
    rdm1b[i,i] = 1.0

for i in range(energya.size):
    for j in range(energya.size):
        rdm2a[i,i,j,j] += 1
        rdm2a[i,j,j,i] -= 1

for i in range(energyb.size):
    for j in range(energyb.size):
        rdm2b[i,i,j,j] += 1
        rdm2b[i,j,j,i] -= 1

for i in range(energya.size):
    for j in range(energyb.size):
        rdm2ab[i,i,j,j] += 1 # only coulomb

for i in range(energyb.size):
    for j in range(energya.size):
        rdm2ba[i,i,j,j] += 1 # only coulomb

den_file = name+'.den'
fspt = open(den_file,'w')
fspt.write('CCIQA\n')
fspt.write('La matriz D es:\n')
for i in range(nmoa):
    for j in range(nmoa):
        fspt.write('%i %i %.4f\n' % ((i+1), (j+1), rdm1a[i,j]))
for i in range(nmob):
    for j in range(nmob):
        fspt.write('%i %i %.4f\n' % ((i+1+nmoa), (j+1+nmoa), rdm1b[i,j]))
fspt.write('La matriz d es:\n')
for i in range(nmoa):
    for j in range(nmoa):
        for k in range(nmoa):
            for l in range(nmoa):
                if (abs(rdm2a[i,j,k,l]) > 1e-12):
                        fspt.write('%i %i %i %i %.4f\n' \
                        % ((i+1), (j+1), (k+1), (l+1), rdm2a[i,j,k,l]))
for i in range(nmob):
    for j in range(nmob):
        for k in range(nmob):
            for l in range(nmob):
                if (abs(rdm2b[i,j,k,l]) > 1e-12):
                        fspt.write('%i %i %i %i %.4f\n' \
                        % ((i+1+nmoa), (j+1+nmoa), (k+1+nmoa), (l+1+nmoa), rdm2b[i,j,k,l]))
for i in range(nmoa):
    for j in range(nmoa):
        for k in range(nmob):
            for l in range(nmob):
                if (abs(rdm2ab[i,j,k,l]) > 1e-12):
                        fspt.write('%i %i %i %i %.4f\n' \
                        % ((i+1), (j+1), (k+1+nmoa), (l+1+nmoa), rdm2ab[i,j,k,l]))
for i in range(nmob):
    for j in range(nmob):
        for k in range(nmoa):
            for l in range(nmoa):
                if (abs(rdm2ba[i,j,k,l]) > 1e-12):
                        fspt.write('%i %i %i %i %.4f\n' \
                        % ((i+1+nmoa), (j+1+nmoa), (k+1), (l+1), rdm2ba[i,j,k,l]))
fspt.close()                    
