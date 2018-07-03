#!/usr/bin/env python

import numpy, sys
sys.path.append('../tools')
from pyscf import gto, scf, dft
from pyscf import lib, ao2mo
import wfn_format, utils
einsum = lib.einsum

name = 'co'

mol = gto.M()
mol.atom = '''
    C 0.0 0.0 0.00
    O 0.0 0.0 1.12
'''
mol.basis = 'aug-cc-pvdz'
mol.verbose = 4
mol.build()

mf = scf.RHF(mol)
mf.conv_tol = 1e-8
mf = utils.smearing(mf, sigma=.1, method='fermi')
ehf = mf.kernel()
eps = 1e-12
lib.logger.info(mf, 'Occ = %s' % mf.mo_occ[mf.mo_occ>eps])
lib.logger.info(mf, 'Entropy = %s' % mf.entropy)
lib.logger.info(mf, 'Free energy = %s' % mf.e_free)
lib.logger.info(mf, 'Zero temperature energy = %s' % ((mf.e_tot+mf.e_free)/2))

c = mf.mo_coeff[:,mf.mo_occ>eps]
energy = mf.mo_energy[mf.mo_occ>eps]
nmo = c.shape[1]
occ = mf.mo_occ[mf.mo_occ>eps]
dm = numpy.diag(occ)

lib.logger.info(mf, '**** Info on MO basis')
s = mol.intor('cint1e_ovlp_sph')
t = mol.intor('cint1e_kin_sph')
h = mf.get_hcore()
s = reduce(numpy.dot, (c.T,s,c))
t = reduce(numpy.dot, (c.T,t,c))
h = reduce(numpy.dot, (c.T,h,c))
enuc = mol.energy_nuc() 
ekin = einsum('ij,ij->',t,dm)
hcore = einsum('ij,ij->',h,dm)
pop = einsum('ij,ij->',s,dm)
lib.logger.info(mf, 'Population : %s' % pop)
lib.logger.info(mf, 'Kinetic energy : %s' % ekin)
lib.logger.info(mf, 'Hcore energy : %s' % hcore)
lib.logger.info(mf, 'Nuclear energy : %s' % enuc)

eri_mo = ao2mo.kernel(mf._eri, c, compact=False)
eri_mo = eri_mo.reshape(nmo,nmo,nmo,nmo)
rdm2_j = einsum('ij,kl->ijkl', dm, dm) 
rdm2_xc = -0.5*einsum('ik,jl->ijkl', dm, dm)
rdm2 = rdm2_j + rdm2_xc
bie1 = einsum('ijkl,ijkl->', eri_mo, rdm2_j)*0.5 
bie2 = einsum('ijkl,ijkl->', eri_mo, rdm2_xc)*0.5
lib.logger.info(mf, 'J energy : %s' % bie1)
lib.logger.info(mf, 'XC energy : %s' % bie2)
lib.logger.info(mf, 'EE energy : %s' % (bie1+bie2))

etot = enuc + hcore + bie1 + bie2
lib.logger.info(mf, 'Total energy : %s' % etot)

with open(name+'.wfn', 'w') as f2:
    wfn_format.write_mo(f2, mol, c, mo_energy=energy, mo_occ=occ)

den_file = name+'.den'
fspt = open(den_file,'w')
fspt.write('CCIQA\n')
fspt.write('La matriz D es:\n')
for i in range(nmo):
    for j in range(nmo):
        fspt.write('%i %i %.16f\n' % ((i+1), (j+1), dm[i,j]))
fspt.write('La matriz d es:\n')
for i in range(nmo):
    for j in range(nmo):
        for k in range(nmo):
            for l in range(nmo):
                if (abs(rdm2[i,j,k,l]) > 1e-8):
                        fspt.write('%i %i %i %i %.10f\n' % \
                        ((i+1), (j+1), (k+1), (l+1), rdm2[i,j,k,l]))
fspt.close()                        

lib.logger.info(mf, '**** Info on AO basis')
nao = mol.nao_nr()
dm = mf.make_rdm1()
s = mol.intor('int1e_ovlp')
t = mol.intor('int1e_kin')
v = mol.intor('int1e_nuc')
eri_ao = ao2mo.restore(1,mf._eri,nao)
eri_ao = eri_ao.reshape(nao,nao,nao,nao)
enuc = mol.energy_nuc() 
ekin = numpy.einsum('ij,ji->',t,dm)
pop = numpy.einsum('ij,ji->',s,dm)
elnuce = numpy.einsum('ij,ji->',v,dm)
lib.logger.info(mf,'Population : %12.6f' % pop)
lib.logger.info(mf,'Kinetic energy : %12.6f' % ekin)
lib.logger.info(mf,'Nuclear Atraction energy : %12.6f' % elnuce)
lib.logger.info(mf,'Nuclear Repulsion energy : %12.6f' % enuc)
bie1 = numpy.einsum('ijkl,ij,kl->',eri_ao,dm,dm)*0.5 # J
bie2 = numpy.einsum('ijkl,il,jk->',eri_ao,dm,dm)*0.25 # XC
pairs1 = numpy.einsum('ij,kl,ij,kl->',dm,dm,s,s)*0.5 # J
pairs2 = numpy.einsum('ij,kl,li,kj->',dm,dm,s,s)*0.25 # XC
pairs = (pairs1 - pairs2)
lib.logger.info(mf,'Coulomb Pairs : %12.6f' % (pairs1))
lib.logger.info(mf,'XC Pairs : %12.6f' % (pairs2))
lib.logger.info(mf,'Pairs : %12.6f' % pairs)
lib.logger.info(mf,'J energy : %12.6f' % bie1)
lib.logger.info(mf,'XC energy : %12.6f' % -bie2)
lib.logger.info(mf,'EE energy : %12.6f' % (bie1-bie2))
etot = enuc + ekin + elnuce + bie1 - bie2
lib.logger.info(mf,'Total energy : %12.6f' % etot)

