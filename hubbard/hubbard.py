#!/usr/bin/env python

import numpy, sys, os
from pyscf.fci import select_ci
from pyscf import gto, scf, symm, dft, fci
from pyscf import ao2mo, mcscf, hci, lib
from pyscf.tools import wfn_format, molden
from pyscf.mcscf import dmet_cas
from pyscf.mcscf import avas

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

def onsite(uparam, nsites):
    result = numpy.zeros((nsites,nsites,nsites,nsites))
    for i in xrange(nsites):
        result[i, i, i, i] = uparam
    return result

def overlap(nsites):
    return numpy.identity(nsites)

name = 'hubbard'
uparam = 4.0
tparam = -6.0
nsites = 6
nelectrons = 6
spin = 0
pbc = False

s = overlap(nsites)
h1 = hop(tparam,nsites,pbc) 
eri = onsite(uparam,nsites)

mol = gto.Mole()
mol.nelectron = nelectrons
mol.verbose = 4
mol.spin = spin
mol.symmetry = 0
mol.charge = 0
mol.incore_anyway = True
mol.build()

mf = scf.RHF(mol)
mf.conv_tol = 1e-8
mf.max_cycle = 150
mf.diis = True
mf.diis_space = 12
mf.get_hcore = lambda *args: h1
mf.get_ovlp = lambda *args: s
mf._eri = ao2mo.restore(8, eri, nsites)
mf = scf.newton(mf)
mf.kernel()
dm = mf.make_rdm1()

print "* HF Results "
print "#######################################"
print "Population"
print "#######################################"
pop = numpy.einsum('ij,ji->i', dm, s)
for ia in range(nsites):
    symb1 = 'H'
    print ia+1, symb1, pop[ia]

print " "
print "#######################################"
print "Delocalization Indexes"
print "#######################################"
pairs2 = numpy.einsum('ij,kl,li,kj->ik',dm,dm,s,s)*0.25 # XC
for ia in range(nsites):
    symb1 = 'H'
    for ib in range(ia+1):
        symb2 = 'H'
        if (ia == ib): 
            factor = 1.0
        if (ia != ib): 
            factor = 2.0
        print ia+1, ib+1, symb1, symb2, 2*factor*pairs2[ia,ib]

mc = mcscf.CASCI(mf, nsites, mol.nelectron)
mc.fcisolver = fci.SCI(mol)
mc.fcisolver.ci_coeff_cutoff = 0.01
mc.fcisolver.select_cutoff = 0.01
mc.fcisolver.conv_tol = 1e-8
mc.fix_spin_(shift=1.2, ss=0.0000)
mc.kernel()

#print('S^2 = %.7f, 2S+1 = %.7f' % mcscf.spin_square(mc))
#for i in range(nsites):
#    ss = fci.spin_op.local_spin(mc.ci, mc.ncas, mc.nelecas, mc.mo_coeff[:,:], overlap, [i])
#    print('local spin for H = %.7f, 2S+1 = %.7f' % (ss))

rdm1, rdm2 = mc.fcisolver.make_rdm12(mc.ci, mc.ncas, mc.nelecas) 
rdm1, rdm2 = mcscf.addons._make_rdm12_on_mo(rdm1, rdm2, mc.ncore, mc.ncas, nsites)
rdm1_mo = rdm1
#print rdm1
rdm2 = rdm2 - numpy.einsum('ij,kl->ijkl',rdm1,rdm1) 
rdm1 = reduce(numpy.dot, (mc.mo_coeff, rdm1, mc.mo_coeff.T))
#print rdm1
rdm2 = numpy.dot(mc.mo_coeff, rdm2.reshape(nsites,-1))
rdm2 = numpy.dot(rdm2.reshape(-1,nsites), mc.mo_coeff.T)
rdm2 = rdm2.reshape(nsites,nsites,nsites,nsites).transpose(2,3,0,1)
rdm2 = numpy.dot(mc.mo_coeff, rdm2.reshape(nsites,-1))
rdm2 = numpy.dot(rdm2.reshape(-1,nsites), mc.mo_coeff.T)
rdm2 = rdm2.reshape(nsites,nsites,nsites,nsites)
rdm2 = -rdm2

print "* FCI results "
print "#######################################"
print "Population"
print "#######################################"
pop = numpy.einsum('ij,ji->i', rdm1, s)
for ia in range(nsites):
    symb1 = 'H'
    print ia+1, symb1, pop[ia]

pairs2 = numpy.einsum('ijkl,ij,kl->ik',rdm2,s,s)*0.5 # XC
print " "
print "#######################################"
print "Delocalization Indexes"
print "#######################################"
for ia in range(nsites):
    symb1 = 'H'
    for ib in range(ia+1):
        symb2 = 'H'
        if (ia == ib): 
            factor = 1.0
        if (ia != ib): 
            factor = 2.0
        print ia+1, ib+1, symb1, symb2, 2*factor*pairs2[ia,ib]

##### Simulate a WFN an a AOM file for EDF  ######        
atms = []
for i in range(nsites):
    atms.append(('H',0.0, 0.0, 0.0))
mol = gto.Mole()
mol.nelectron = nelectrons
mol.verbose = 0
mol.spin = spin
mol.symmetry = 0
mol.charge = 0
mol.incore_anyway = True
mol.basis = 'sto-3g'
mol.atom = atms
mol.build()
nmo = mc.ncore + mc.ncas
natocc, natorb = numpy.linalg.eigh(-rdm1)
for i, k in enumerate(numpy.argmax(abs(natorb), axis=0)):
    if natorb[k,i] < 0:
        natorb[:,i] *= -1
natorb = numpy.dot(mc.mo_coeff[:,:nmo], natorb)
natocc = -natocc
wfn_file = name + '.wfn'
with open(wfn_file, 'w') as f2:
    wfn_format.write_mo(f2, mol, natorb, mo_occ=natocc)
    wfn_format.write_coeff(f2, mol, mc.mo_coeff[:,:nmo])
    #wfn_format.write_ci(f2, mc.ci, mc.ncas, mc.nelecas, ncore=mc.ncore)
    #hci.to_fci_wfn_gs_1(f2, mc.ci, mc.ncas, mc.nelecas, root=0, ncore=mc.ncore)
    #wfn_format.write_ci(f2, select_ci.to_fci(mc.ci,mc.ncas,mc.nelecas), mc.ncas, mc.nelecas, ncore=mc.ncore)
aom_file = name + '.wfn.aom'
with open(aom_file, 'w') as f2:
    for k in range(nsites): # Over atoms == over primitives
        f2.write("%5d <=== AOM within this center\n" % (k+1))
        ij = 0
        for i in range(nsites):
            for j in range(i+1):
                f2.write(' %16.10f' % (mc.mo_coeff[k,i]*mc.mo_coeff[k,j]))
                ij += 1
                if (ij%6 == 0):
                    f2.write("\n")
        f2.write("\n")
#1 format (6(1x,e16.10)) --> TODO: write modulo 6
