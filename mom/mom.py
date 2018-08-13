#!/usr/bin/env python

import numpy, sys
sys.path.append('../tools')
from pyscf import gto, scf, dft
from pyscf.tools import molden
import imom

name = 'i-mom'

mol = gto.Mole()
mol.verbose = 4
mol.atom = '''
c   1.217739890298750 -0.703062453466927  0.000000000000000
h   2.172991468538160 -1.254577209307266  0.000000000000000
c   1.217739890298750  0.703062453466927  0.000000000000000
h   2.172991468538160  1.254577209307266  0.000000000000000
c   0.000000000000000  1.406124906933854  0.000000000000000
h   0.000000000000000  2.509154418614532  0.000000000000000
c  -1.217739890298750  0.703062453466927  0.000000000000000
h  -2.172991468538160  1.254577209307266  0.000000000000000
c  -1.217739890298750 -0.703062453466927  0.000000000000000
h  -2.172991468538160 -1.254577209307266  0.000000000000000
c   0.000000000000000 -1.406124906933854  0.000000000000000
h   0.000000000000000 -2.509154418614532  0.000000000000000
'''
mol.basis = 'def2-svpd'
mol.symmetry = 0
mol.build()

a = dft.UKS(mol)
a.xc = 'm06-2x'
a.scf()

mo0 = a.mo_coeff
occ = a.mo_occ

# Assign initial occupation pattern
occ[0][20]=0 # this excited state is originated from HOMO(alpha) -> LUMO(alpha)
occ[0][21]=1 # it is still a singlet state
occ[1][20]=0 # this excited state is originated from HOMO(beta) -> LUMO(beta)
occ[1][21]=1 # it is still a singlet state

mf = dft.UKS(mol)
mf.xc = 'm06-2x'
dm_u = mf.make_rdm1(mo0, occ)
mf = scf.addons.mom_occ(mf, mo0, occ)
mf.scf(dm_u)

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

print('----------------UHF calculation----------------')
print('Excitation energy(UKS): %.3g eV' % ((mf.e_tot - a.e_tot)*27.211))
print('Alpha electron occpation pattern of excited state(UKS) : %s' %(mf.mo_occ[0]))
print(' Beta electron occpation pattern of excited state(UKS) : %s' %(mf.mo_occ[1]))
