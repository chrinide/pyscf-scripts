#!/usr/bin/env python

import numpy
import scipy.linalg
import os
from pyscf import scf
from pyscf import gto
from pyscf import mcscf, fci, cc
from pyscf import tools
from pyscf import ao2mo
from pyscf import symm
from pyscf.tools import molden, wfn_format

name = 'h2_ccsd'
mol = gto.Mole()
mol.verbose = 4
mol.atom = '''
H 0.0 0.0 0.0
H 0.0 0.0 0.75
    '''
mol.basis = 'cc-pvtz'
mol.symmetry = 0
mol.spin = 0
mol.charge = 0
mol.build()

mf = scf.RHF(mol)
ehf = mf.kernel()

mcc = cc.ccsd.CC(mf)
#mcc.frocen = 0
mcc.direct = 1
mcc.conv_tol = 1e-9
mcc.conv_tol_normt = 1e-8
mcc.kernel()
mcc.solve_lambda()
rdm1 = mcc.make_rdm1()
rdm2 = mcc.make_rdm2()

#######################################
def add_inactive_space_to_rdm(mol, nsize, one_pdm, two_pdm):
    '''If a CASSCF calculation has been done, the final RDMs will
    not contain the doubly occupied inactive orbitals. This function will add
    them and return the full density matrices.
    '''

    # Find number of inactive electrons by taking the number of electrons
    # as the trace of the 1RDM, and subtracting from the total number of
    # electrons
    ninact = (mol.nelectron - int(round(numpy.trace(one_pdm)))) / 2
    norb = nsize
    nsizerdm = one_pdm.shape[0]

    one_pdm_ = numpy.zeros( (norb, norb) )
    # Add the core first.
    for i in range(ninact):
        one_pdm_[i,i] = 2.0

    # Add the rest of the density matrix.
    one_pdm_[ninact:ninact+nsizerdm,ninact:ninact+nsizerdm] = one_pdm[:,:]

    two_pdm_ = numpy.zeros( (norb, norb, norb, norb) )
    
    # Add on frozen core contribution, assuming that the inactive orbitals are
    # doubly occupied.
    for i in range(ninact):
        for j in range(ninact):
            two_pdm_[i,i,j,j] +=  4.0
            two_pdm_[i,j,j,i] += -2.0

    # Inactve-Active elements.
    for p in range(ninact):
        for r in range(ninact,ninact+nsizerdm):
            for s in range(ninact,ninact+nsizerdm):
                two_pdm_[p,p,r,s] += 2.0*one_pdm_[r,s]
                two_pdm_[r,s,p,p] += 2.0*one_pdm_[r,s]
                two_pdm_[p,r,s,p] -= one_pdm_[r,s]
                two_pdm_[r,p,p,s] -= one_pdm_[r,s]

    # Add active space.
    two_pdm_[ninact:ninact+nsizerdm,ninact:ninact+nsizerdm, \
             ninact:ninact+nsizerdm,ninact:ninact+nsizerdm] = \
             two_pdm[:,:]

    return one_pdm_, two_pdm_

nao = mol.nao_nr()
rdm1, rdm2 = add_inactive_space_to_rdm(mol, nao, rdm1, rdm2)
#######################################

rdm2 = rdm2 - numpy.einsum('ij,kl->ijkl',rdm1,rdm1) 
rdm1 = reduce(numpy.dot, (mf.mo_coeff, rdm1, mf.mo_coeff.T))
rdm2 = numpy.dot(mf.mo_coeff, rdm2.reshape(nao,-1))
rdm2 = numpy.dot(rdm2.reshape(-1,nao), mf.mo_coeff.T)
rdm2 = rdm2.reshape(nao,nao,nao,nao).transpose(2,3,0,1)
rdm2 = numpy.dot(mf.mo_coeff, rdm2.reshape(nao,-1))
rdm2 = numpy.dot(rdm2.reshape(-1,nao), mf.mo_coeff.T)
rdm2 = rdm2.reshape(nao,nao,nao,nao)
rdm2 = -rdm2

s = mol.intor('cint1e_ovlp_sph')
t = mol.intor('cint1e_kin_sph')
eri_ao = mol.intor('cint2e_sph')
eri_ao = eri_ao.reshape(nao,nao,nao,nao)

label = mol.spheric_labels(False)

mull_file = name+'.mullout'
fspt = open(mull_file,'w')
fspt.write('\n')
fspt.write('#########################################\n')
fspt.write('#########################################\n')   
fspt.write('Mulliken Analysis for point : %s \n' % (name))
fspt.write('#########################################\n')
fspt.write('#########################################\n')
fspt.write('\n')

fspt.write('#################\n')
fspt.write('Monocentric terms\n')
fspt.write('#################\n')
pop = numpy.einsum('ij,ij->i', rdm1,s)
kin = numpy.einsum('ij,ij->i', rdm1,t) 
chg1 = numpy.zeros(mol.natm)
kinatm = numpy.zeros(mol.natm)
qq1 = numpy.zeros(mol.natm)
eself = numpy.zeros((mol.natm,mol.natm))
for i, s1 in enumerate(label):
    chg1[s1[0]] += pop[i]
    kinatm[s1[0]] += kin[i]
for ia in range(mol.natm):
    symb = mol.atom_symbol(ia)
    qq1[ia] = mol.atom_charge(ia)-chg1[ia]
    fspt.write('Pop, Q, K of %d %s = %12.6f %12.6f %12.6f \n' % (ia, symb, chg1[ia], qq1[ia], kinatm[ia]))
fspt.write('Sum of charges : %12.6f \n' % sum(chg1))
fspt.write('\n')

fspt.write('###############\n')
fspt.write('Bicentric terms\n')
fspt.write('###############\n')
#### Nuclear-electron matrix
charges = mol.atom_charges()
coords = mol.atom_coords()
vpot = numpy.zeros((mol.natm,mol.natm))
for i in range(mol.natm):
    q = charges[i]
    r = coords[i]
    mol.set_rinv_origin(coords[i])
    v = mol.intor('cint1e_rinv_sph') * -q
    vtmp = numpy.einsum('ij,ij->i', rdm1,v) 
    for j, vl in enumerate(label):
        vpot[i,vl[0]] += vtmp[j]
#### Nuclear-Nuclear matrix
rr = numpy.dot(coords, coords.T)
rd = rr.diagonal()
rr = rd[:,None] + rd - rr*2
rr[numpy.diag_indices_from(rr)] = 1e-60
r = numpy.sqrt(rr)
qq = charges[:,None] * charges[None,:]
qq[numpy.diag_indices_from(qq)] = 0
enuc = qq/r * 0.5
### Bielectronic terms
bie1 = numpy.einsum('ijkl,ij,kl->ik',eri_ao,rdm1,rdm1)*0.5
bie2 = numpy.einsum('ijkl,ijkl->ik',eri_ao,rdm2)*0.5
pairs1 = numpy.einsum('ij,kl,ij,kl->ik',rdm1,rdm1,s,s)*0.5 # J
pairs2 = numpy.einsum('ijkl,ij,kl->ik',rdm2,s,s)*0.5 # XC
pop = (pairs1 - pairs2)
##########################
chg = numpy.zeros((mol.natm,mol.natm))
chg1 = numpy.zeros((mol.natm,mol.natm))
chg2 = numpy.zeros((mol.natm,mol.natm))
je = numpy.zeros((mol.natm,mol.natm))
xc = numpy.zeros((mol.natm,mol.natm))
inter = numpy.zeros((mol.natm,mol.natm))
clasica = numpy.zeros((mol.natm,mol.natm))
for i, s1 in enumerate(label):
    for j, s2 in enumerate(label):
        factor = 1.0
        chg[s1[0],s2[0]] += pop[i,j]*factor
        chg1[s1[0],s2[0]] += pairs1[i,j]*factor
        chg2[s1[0],s2[0]] += pairs2[i,j]*factor
        je[s1[0],s2[0]] += bie1[i,j]*factor
        xc[s1[0],s2[0]] += bie2[i,j]*factor
for  ia in range(mol.natm):
    eself[ia,ia] = kinatm[ia] + je[ia,ia] + vpot[ia,ia] -xc[ia,ia] 
check = 0        
checkii = 0
checkij = 0
eecheck = 0        
jcheck = 0
xccheck = 0
enuccheck = 0
vpotcheck = 0
checkj = 0
checkxc = 0
for ia in range(mol.natm):
    symb1 = mol.atom_symbol(ia)
    for ib in range(ia+1):
        symb2 = mol.atom_symbol(ib)
        if (ia == ib): 
            factor = 1.0
            vpotcheck = vpotcheck + vpot[ia,ib]
            checkii = checkii + chg[ia,ib]
            check = check + chg[ia,ib]
            eecheck = eecheck + (je[ia,ib]-xc[ia,ib])
            jcheck = jcheck + je[ia,ib]
            xccheck = xccheck - xc[ia,ib]
            checkj = checkj + chg1[ia,ib]
            checkxc = checkxc + chg2[ia,ib]
            clasica[ia,ib] = vpot[ia,ib] + vpot[ib,ia] + factor*je[ia,ib]
            clasica[ib,ia] = clasica[ia,ib]
            inter[ia,ib] = clasica[ia,ib] - factor*xc[ia,ib] 
            inter[ib,ia] = inter[ia,ib]
        if (ia != ib): 
            factor = 2.0
            checkj = checkj + factor*chg1[ia,ib]
            checkxc = checkxc + factor*chg2[ia,ib]
            vpotcheck = vpotcheck + vpot[ia,ib] + vpot[ib,ia]
            checkij = checkij + factor*chg[ia,ib]
            check = check + factor*chg[ia,ib]
            eecheck = eecheck + factor*(je[ia,ib]-xc[ia,ib])
            jcheck = jcheck + factor*je[ia,ib]
            xccheck = xccheck - factor*xc[ia,ib]
            enuc[ia,ib] = enuc[ia,ib]*factor
            enuccheck = enuccheck + enuc[ia,ib]
            clasica[ia,ib] = enuc[ia,ib] + vpot[ia,ib] + vpot[ib,ia] + factor*je[ia,ib]
            clasica[ib,ia] = clasica[ia,ib]
            inter[ia,ib] = clasica[ia,ib] - factor*xc[ia,ib] 
            inter[ib,ia] = inter[ia,ib]
        fspt.write('Lambda-Delta, Pairs, Self, N-N, N-E, E-N, J, XC, Cl, Int of  %d %d %s %s = \
        %12.6f %12.6f %12.6f %12.6f %12.6f %12.6f %12.6f %12.6f %12.6f %12.6f \n' % \
        (ia, ib, symb1, symb2, 2*factor*chg2[ia,ib], chg[ia,ib], eself[ia,ib],  enuc[ia,ib], \
        vpot[ia,ib], vpot[ib,ia], factor*je[ia,ib], -factor*xc[ia,ib], clasica[ia,ib], inter[ia,ib]))

fspt.write('\n')
fspt.write('##############\n')
fspt.write('Sum rules test\n')
fspt.write('##############\n')
fspt.write('Total Coulomb pairs : %12.6f\n' % checkj)
fspt.write('Total XC pairs : %12.6f\n' % checkxc)
fspt.write('Total intra pairs : %12.6f\n' % checkii)
fspt.write('Total inter pairs : %12.6f\n' % checkij)
fspt.write('Total pairs : %12.6f\n' % check)
fspt.write('Total N-N : %12.6f\n' % enuccheck)
fspt.write('Total N-E + E-N : %12.6f\n' % vpotcheck)
fspt.write('Total J energy : %12.6f\n' % jcheck)
fspt.write('Total XC energy : %12.6f\n' % xccheck)
fspt.write('Total energy is : %12.6f\n' % (sum(kin)+jcheck+xccheck+enuccheck+vpotcheck))
fspt.write('\n')
fspt.close()

##############################################################################
# Print a file for xpyscf
##############################################################################
resume_file = name+'.info'
fspt = open(resume_file,'w')
fspt.write('%s \n' % __file__)
fspt.write('C1\n')
fspt.write('%i \n' % mol.natm)
fspt.write('%i \n' % mol.natm)
coords = mol.atom_coords()
for ia in range(mol.natm):
    symb = mol.atom_symbol(ia)
    fspt.write('%12.6f  %12.6f  %12.6f %s\n' % (coords[ia][0], coords[ia][1], coords[ia][2], symb))
#########################################################
for ia in range(mol.natm): # kinetic
    fspt.write('%12.6f \n' % kinatm[ia])
for ia in range(mol.natm): # Potential energy
    fspt.write('%12.6f \n' % (0.0))
for ia in range(mol.natm): # EE
    fspt.write('%12.6f \n' % (je[ia,ia]-xc[ia,ia]))
for ia in range(mol.natm): # Coul
    fspt.write('%12.6f \n' % (je[ia,ia]))
for ia in range(mol.natm): # XC
    fspt.write('%12.6f \n' % (-xc[ia,ia]))
for ia in range(mol.natm): # Self NE
    fspt.write('%12.6f \n' % (vpot[ia,ia]))
for ia in range(mol.natm): # Net energy
    fspt.write('%12.6f \n' % (eself[ia,ia]))
for ia in range(mol.natm): # Monocentric Interaction energy ?
    fspt.write('%12.6f \n' % (inter[ia,ia]))
for ia in range(mol.natm): # Additive energy
    fspt.write('%12.6f \n' % (0.0))
for ia in range(mol.natm): # Effective energy
    fspt.write('%12.6f \n' % (0.0))
for ia in range(mol.natm): # Corr
    fspt.write('%12.6f \n' % (0.0))
#########################################################
for ia in range(mol.natm): # Bicentric Interaction energy
    for ib in range(mol.natm):
        if (ia != ib):
            fspt.write('%12.6f %12.6f %12.6f %12.6f %12.6f \n' % (enuc[ia,ib], vpot[ia,ib], vpot[ib,ia], 2.0*je[ia,ib]-2.0*xc[ia,ib], inter[ia,ib]))
for ia in range(mol.natm): 
    for ib in range(mol.natm):
        if (ia != ib):
            fspt.write('%12.6f %12.6f \n' % (2.0*je[ia,ib],-2.0*xc[ia,ib]))
for ia in range(mol.natm): 
    for ib in range(mol.natm):
        if (ia != ib):
            fspt.write('%12.6f \n' % (0.0))
#########################################################
for ia in range(mol.natm):
    fspt.write('%12.6f \n' % (qq1[ia]))
#########################################################
fspt.close()
