#!/usr/bin/env python

import numpy
from pyscf import gto, scf, dft
from pyscf.tools import wfn_format

name = 'h2o'

mol = gto.Mole()
mol.symmetry = 0
mol.verbose = 4
mol.atom = [
    ["O" , (0. , 0.     , 0.)],
    ["H" , (0. , -0.757 , 0.587)],
    ["H" , (0. , 0.757  , 0.587)] ]
mol.basis = {"H": '6-31g',
             "O": '6-31g',}
mol.build()

a = scf.UHF(mol)
a.scf()
mo0 = a.mo_coeff
occ = a.mo_occ

# Assign initial occupation pattern
occ[0][4]=0  # this excited state is originated from HOMO(alpha) -> LUMO(alpha)
occ[0][5]=1  # it is still a singlet state

# New SCF caculation 
mf = scf.UHF(mol)
# Construct new dnesity matrix with new occpuation pattern
dm_u = mf.make_rdm1(mo0, occ)
# Apply mom occupation principle
mf = scf.addons.mom_occ(mf, mo0, occ)
# Start new SCF with new density matrix
mf.scf(dm_u)

coeffa = mf.mo_coeff[0][:,mf.mo_occ[0]>0]
energya = mf.mo_energy[0][mf.mo_occ[0]>0]
occa = mf.mo_occ[0][mf.mo_occ[0]>0]

coeffb = mf.mo_coeff[1][:,mf.mo_occ[1]>0]
energyb = mf.mo_energy[1][mf.mo_occ[1]>0]
occb = mf.mo_occ[1][mf.mo_occ[1]>0]

with open(name+'.wfn', 'w') as f2:
    wfn_format.write_mo(f2, mol, coeffa, mo_energy=energya, mo_occ=occa)
    wfn_format.write_mo(f2, mol, coeffb, mo_energy=energyb, mo_occ=occb)
    
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
        fspt.write('%i %i %.16f\n' % ((i+1), (j+1), rdm1a[i,j]))
for i in range(nmob):
    for j in range(nmob):
        fspt.write('%i %i %.16f\n' % ((i+1+nmoa), (j+1+nmoa), rdm1b[i,j]))
fspt.write('La matriz d es:\n')
for i in range(nmoa):
    for j in range(nmoa):
        for k in range(nmoa):
            for l in range(nmoa):
                if (abs(rdm2a[i,j,k,l]) > 1e-12):
                        fspt.write('%i %i %i %i %.16f\n' % \
                        ((i+1), (j+1), (k+1), (l+1), rdm2a[i,j,k,l]))
for i in range(nmob):
    for j in range(nmob):
        for k in range(nmob):
            for l in range(nmob):
                if (abs(rdm2b[i,j,k,l]) > 1e-12):
                        fspt.write('%i %i %i %i %.16f\n' % \
                        ((i+1+nmoa), (j+1+nmoa), (k+1+nmoa), (l+1+nmoa), rdm2b[i,j,k,l]))
for i in range(nmoa):
    for j in range(nmoa):
        for k in range(nmob):
            for l in range(nmob):
                if (abs(rdm2ab[i,j,k,l]) > 1e-12):
                        fspt.write('%i %i %i %i %.16f\n' % \
                        ((i+1), (j+1), (k+1+nmoa), (l+1+nmoa), rdm2ab[i,j,k,l]))
for i in range(nmob):
    for j in range(nmob):
        for k in range(nmoa):
            for l in range(nmoa):
                if (abs(rdm2ba[i,j,k,l]) > 1e-12):
                        fspt.write('%i %i %i %i %.16f\n' % \
                        ((i+1+nmoa), (j+1+nmoa), (k+1), (l+1), rdm2ba[i,j,k,l]))
fspt.close()                    

