#!/usr/bin/env python

import numpy, sys
from pyscf import gto, scf, ao2mo

mol = gto.Mole()
mol.basis = {'H':'cc-pvdz','C':'stuttgart-dz'}
mol.ecp = 'stuttgart-dz'
mol.atom = '''
C  0.0000  0.0000  0.0000
H  0.6276  0.6276  0.6276
H  0.6276 -0.6276 -0.6276
H -0.6276  0.6276 -0.6276
H -0.6276 -0.6276  0.6276
'''
mol.charge = 0
mol.spin = 0
mol.symmetry = 1
mol.verbose = 4
mol.build()

mf = scf.RHF(mol)
ehf = mf.kernel()

nao = mol.nao_nr()
dm = mf.make_rdm1()
mo = mf.mo_coeff
c = numpy.eye(mol.nao_nr())
print('The HF energy is : %12.6f' % ehf)

s = mol.intor_symmetric('int1e_ovlp')
t = mol.intor_symmetric('int1e_kin')
v = mol.intor_symmetric('int1e_nuc')
ecp = mol.intor_symmetric('ECPscalar')
eecp = numpy.einsum('ij,ji->',ecp,dm) 
enuc = mol.energy_nuc() 
ekin = numpy.einsum('ij,ji->',t,dm)
pop = numpy.einsum('ij,ji->',s,dm)
elnuce = numpy.einsum('ij,ji->',v,dm)
print('Population : %12.6f' % pop)
print('ECP : %12.6f' % eecp)
print('Kinetic energy : %12.6f' % ekin)
print('Nuclear Atraction energy : %12.6f' % elnuce)
print('Nuclear Repulsion energy : %12.6f' % enuc)

eri_ao = ao2mo.restore(1, mf._eri, nao)
eri_ao = eri_ao.reshape(nao,nao,nao,nao)
bie1 = numpy.einsum('ijkl,ij,kl',eri_ao,dm,dm)*0.5 # J
bie2 = numpy.einsum('ijkl,il,jk',eri_ao,dm,dm)*0.25 # XC
pairs1 = numpy.einsum('ij,kl,ij,kl->',dm,dm,s,s)*0.5 # J
pairs2 = numpy.einsum('ij,kl,li,kj->',dm,dm,s,s)*0.25 # XC
pairs = (pairs1 - pairs2)
print('Coulomb Pairs : %12.6f' % (pairs1))
print('XC Pairs : %12.6f' % (pairs2))
print('Pairs : %12.6f' % pairs)
print('J energy : %12.6f' % bie1)
print('XC energy : %12.6f' % -bie2)
print('EE energy : %12.6f' % (bie1-bie2))

etot = enuc + ekin + elnuce + bie1 - bie2 + eecp
print('Total energy : %12.6f' % etot)

##############################################################################
# Mulliken like energy partition in a IQA stile
##############################################################################
label = mol.spheric_labels(False)
print('\n#########################')
print('Mulliken energy partition')
print('#########################\n')
##############################################################################
##############################################################################
print('Monocentric terms')
print('#################')
pop = numpy.einsum('ij->i', (dm)*s)
kin = numpy.einsum('ij->i', (dm)*t) 
ecp = numpy.einsum('ij->i', (dm)*ecp) 
ecpatm = numpy.zeros(mol.natm)
aecp = numpy.zeros(mol.natm)
chg1 = numpy.zeros(mol.natm)
qq1 = numpy.zeros(mol.natm)
kinatm = numpy.zeros(mol.natm)
eself = numpy.zeros((mol.natm,mol.natm))
for i, s1 in enumerate(label):
    chg1[s1[0]] += pop[i]
    kinatm[s1[0]] += kin[i]
    ecpatm[s1[0]] += ecp[i]
for ia in range(mol.natm):
    symb = mol.atom_symbol(ia)
    qq1[ia] = mol.atom_charge(ia)-chg1[ia]
    print('Pop, Q, K of %d %s = %12.6f %12.6f %12.6f' % (ia, symb, chg1[ia], qq1[ia], kinatm[ia]))
print('\nSum rules test')
print('##############')
print('Sum of charges : %12.6f' % sum(chg1))
print('Sum of kin : %12.6f' % sum(kin))
##############################################################################
##############################################################################
print('\nBicentric terms')
print('###############')
#### Nuclear-electron matrix
charges = mol.atom_charges()
coords = mol.atom_coords()
vpot = numpy.zeros((mol.natm,mol.natm))
for i in range(mol.natm):
    q = charges[i]
    r = coords[i]
    mol.set_rinv_origin(coords[i])
    v = mol.intor('cint1e_rinv_sph') * -q
    v = reduce(numpy.dot, (c.T,v,c))
    vtmp = numpy.einsum('ij->i', (dm)*v) 
    for j, vl in enumerate(label):
        vpot[i,vl[0]] += vtmp[j]
#### Nuclear-Nuclear matrix
charges = mol.atom_charges()
coords = mol.atom_coords()
rr = numpy.dot(coords, coords.T)
rd = rr.diagonal()
rr = rd[:,None] + rd - rr*2
rr[numpy.diag_indices_from(rr)] = 1e-60
r = numpy.sqrt(rr)
qq = charges[:,None] * charges[None,:]
qq[numpy.diag_indices_from(qq)] = 0
enuc = qq/r * 0.5
### Pairs,J,XC
pairs1 = numpy.einsum('ij,kl,ij,kl->ik',dm,dm,s,s)*0.5 # J
pairs2 = numpy.einsum('ij,kl,li,kj->ik',dm,dm,s,s)*0.25 # XC
pop = (pairs1 - pairs2)
bie1 = numpy.einsum('ij,kl,ijkl->ik',dm,dm,eri_ao)*0.5
bie2 = numpy.einsum('il,kj,ijkl->ik',dm,dm,eri_ao)*0.25
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
    eself[ia,ia] = kinatm[ia] + je[ia,ia] + vpot[ia,ia] -xc[ia,ia] + ecpatm[ia]
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
            enuc[ib,ia] = enuc[ia,ib]
            enuccheck = enuccheck + enuc[ia,ib]
            clasica[ia,ib] = enuc[ia,ib] + vpot[ia,ib] + vpot[ib,ia] + factor*je[ia,ib]
            clasica[ib,ia] = clasica[ia,ib]
            inter[ia,ib] = clasica[ia,ib] - factor*xc[ia,ib]
            inter[ib,ia] = inter[ia,ib]
        print('Lambda-Delta, Pairs, Self, N-N, N-E, E-N, J, XC, Cl, Int of  %d %d %s %s = \
        %12.6f %12.6f %12.6f %12.6f %12.6f %12.6f %12.6f %12.6f %12.6f %12.6f' % \
        (ia, ib, symb1, symb2, 2*factor*chg2[ia,ib], chg[ia,ib], eself[ia,ib], enuc[ia,ib], \
        vpot[ia,ib], vpot[ib,ia], factor*je[ia,ib], -factor*xc[ia,ib], clasica[ia,ib], inter[ia,ib]))

print('\n##############')
print('Sum rules test')
print('##############')
print('Total Coulomb pairs : %12.6f' % checkj)
print('Total XC pairs : %12.6f' % checkxc)
print('Total intra pairs : %12.6f' % checkii)
print('Total inter pairs : %12.6f' % checkij)
print('Total pairs : %12.6f' % check)
print('Total N-N : %12.6f' % enuccheck)
print('Total N-E + E-N : %12.6f' % vpotcheck)
print('Total J energy : %12.6f' % jcheck)
print('Total XC energy : %12.6f' % xccheck)
##############################################################################
##############################################################################
print('\nTotal energy is : %12.6f' % (sum(kin+ecp)+jcheck+xccheck+enuccheck+vpotcheck))

##############################################################################
# Print a file for xpyscf
##############################################################################
resume_file = 'data-from-pyscf'
fspt = open(resume_file,'w')
fspt.write('%s \n' % __file__)
fspt.write('C1\n')
fspt.write('%i \n' %mol.natm)
fspt.write('%i \n' %mol.natm)
coords = mol.atom_coords()
for ia in range(mol.natm):
    symb = mol.atom_symbol(ia)
    fspt.write('%12.6f  %12.6f  %12.6f %s\n' % (coords[ia][0],coords[ia][1], coords[ia][2],symb))
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
#for ia in range(mol.natm): # Monocentric Interaction energy ?
#    fspt.write('%12.6f \n' % (inter[ia,ia]))
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
