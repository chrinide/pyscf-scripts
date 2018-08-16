#!/usr/bin/env python

import numpy, sys
from pyscf import gto, scf, ao2mo
from pyscf.tools import wfn_format

name = 'h2o'

mol = gto.Mole()
mol.basis = 'cc-pvtz'
mol.atom = open('../geom/h2o.xyz').read()
mol.verbose = 4
mol.spin = 0
mol.symmetry = 1
mol.charge = 0
mol.build()

mf = scf.RHF(mol)
ehf = mf.kernel()

nmo = mol.nelectron/2
coeff = mf.mo_coeff[:,mf.mo_occ>0]
occ = mf.mo_occ[mf.mo_occ>0]
#
rdm1 = numpy.diag(occ)
x, y = rdm1.nonzero()
nonzero = zip(x,y)
ifile = name + '.rdm1'
with open(ifile, 'w') as f2:
    for i, j in nonzero:
        f2.write('%i %i %.16f\n' % (i, j, rdm1[i,j]))
#        
rdm2_xc = numpy.zeros((nmo,nmo,nmo,nmo))
for i in range(nmo):
    for j in range(nmo):
        rdm2_xc[i,j,j,i] -= 2
x, y, z, w = rdm2_xc.nonzero()
nonzero = zip(x,y,z,w)
ifile = name + '.rdm2_xc'
with open(ifile, 'w') as f2:
    for i, j, k, l in nonzero:
        f2.write('%i %i %i %i %.16f\n' % (i, j, k, l, rdm2_xc[i,j,k,l]))

wfn_file = name + '.wfn'
with open(wfn_file, 'w') as f2:
    wfn_format.write_mo(f2, mol, coeff, occ)

##############################################################################
s = mol.intor('cint1e_ovlp_sph')
s = reduce(numpy.dot, (coeff.T,s,coeff))
x, y = s.nonzero()
nonzero = zip(x,y)
ifile = name + '.overlap'
with open(ifile, 'w') as f2:
    for i, j in nonzero:
        f2.write('%i %i %.16f\n' % (i, j, s[i,j]))
#       
t = mol.intor('cint1e_kin_sph')
t = reduce(numpy.dot, (coeff.T,t,coeff))
x, y = t.nonzero()
nonzero = zip(x,y)
ifile = name + '.kinetic'
with open(ifile, 'w') as f2:
    for i, j in nonzero:
        f2.write('%i %i %.16f\n' % (i, j, t[i,j]))
#        
v = mol.intor('cint1e_nuc_sph')
v = reduce(numpy.dot, (coeff.T,v,coeff))
x, y = v.nonzero()
nonzero = zip(x,y)
ifile = name + '.nucelec'
with open(ifile, 'w') as f2:
    for i, j in nonzero:
        f2.write('%i %i %.16f\n' % (i, j, v[i,j]))
#        
enuc = mol.energy_nuc() 
ekin = numpy.einsum('ij,ji->',t,rdm1)
pop = numpy.einsum('ij,ji->',s,rdm1)
elnuce = numpy.einsum('ij,ji->',v,rdm1)
print('Population : %12.6f' % pop)
print('Kinetic energy : %12.6f' % ekin)
print('Nuclear Atraction energy : %12.6f' % elnuce)
print('Nuclear Repulsion energy : %12.6f' % enuc)
##############################################################################

##############################################################################
eri_mo = ao2mo.kernel(mf._eri, coeff, compact=False)
eri_mo = eri_mo.reshape(nmo,nmo,nmo,nmo)
x, y, z, w = eri_mo.nonzero()
nonzero = zip(x,y,z,w)
ifile = name + '.eri'
with open(ifile, 'w') as f2:
    for i, j, k, l in nonzero:
        f2.write('%i %i %i %i %.16f\n' % (i, j, k, l, eri_mo[i,j,k,l]))
#        
bie1 = numpy.einsum('ijkl,ij,kl',eri_mo,rdm1,rdm1)*0.5 # J
bie2 = numpy.einsum('ijkl,ijkl',eri_mo,rdm2_xc)*0.5 # XC
pairs1 = numpy.einsum('ij,kl,ij,kl->',rdm1,rdm1,s,s) # J
pairs2 = numpy.einsum('ijkl,ij,kl->',rdm2_xc,s,s) # XC
pairs = (pairs1 + pairs2)
print('Coulomb Pairs : %12.6f' % (pairs1))
print('XC Pairs : %12.6f' % (pairs2))
print('Pairs : %12.6f' % pairs)
print('Should be : %i' % (mol.nelectron*(mol.nelectron-1)))
print('J energy : %12.6f' % bie1)
print('XC energy : %12.6f' % bie2)
print('EE energy : %12.6f' % (bie1+bie2))
##############################################################################

etot = enuc + ekin + elnuce + bie1 + bie2
print('Total energy : %12.6f' % etot)
