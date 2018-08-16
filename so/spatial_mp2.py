#!/usr/bin/env python

import numpy
from functools import reduce
from pyscf import scf, gto, mp, lib, ao2mo
from pyscf.tools import molden
einsum = lib.einsum

mol = gto.Mole()
mol.basis = 'cc-pvdz'
mol.atom = '''
C  0.0000  0.0000  0.0000
H  0.6276  0.6276  0.6276
H  0.6276 -0.6276 -0.6276
H -0.6276  0.6276 -0.6276
H -0.6276 -0.6276  0.6276
'''
mol.spin = 0
mol.verbose = 4
mol.build()

mf = scf.UHF(mol)
ehf = mf.kernel()
nao,nmo = mf.mo_coeff[0].shape

ncore = (1,1)
nelec = mol.nelectron
nelecb = nelec//2
neleca = nelec - nelecb
ncorea = ncore[0]
ncoreb = ncore[1]
nocca = neleca - ncorea
nvira = nmo - nocca - ncorea
noccb = nelecb - ncoreb
nvirb = nmo - noccb - ncoreb
nocc = nocca + noccb
nvir = nvira + nvirb
lib.logger.info(mf,"* Core orbitals: %d %d" % ncore)
lib.logger.info(mf,"* Virtual orbitals: %d %d" % (nvira,nvirb))

coa = mf.mo_coeff[0][:,ncorea:ncorea+nocca] 
cob = mf.mo_coeff[1][:,ncoreb:ncoreb+noccb]  
cva = mf.mo_coeff[0][:,ncorea+nocca:] 
cvb = mf.mo_coeff[1][:,ncoreb+noccb:]  
eoa = mf.mo_energy[0][ncorea:ncorea+nocca] 
eob = mf.mo_energy[1][ncoreb:ncoreb+noccb]  
eva = mf.mo_energy[0][ncorea+nocca:] 
evb = mf.mo_energy[1][ncoreb+noccb:]  

# Transform the alpha part
eri = ao2mo.general(mol, (coa,cva,coa,cva))
eri = eri.reshape(nocca,nvira,nocca,nvira)
e_denom = 1.0/(eoa.reshape(-1, 1, 1, 1) - eva.reshape(-1, 1, 1) + \
               eoa.reshape(-1, 1) - eva)
rdm2_ss_aa = numpy.zeros((nocca,nvira,nocca,nvira))
rdm2_ss_aa = einsum('iajb,iajb->iajb', eri, e_denom)
rdm2_ss_aa -= einsum('ibja,iajb->iajb', eri, e_denom)
echeck_aa = einsum('iajb,iajb->', eri, rdm2_ss_aa)*0.5
print('E(MP2 SS AA) is %12.8f' % echeck_aa)

# Transform the beta part
eri = ao2mo.general(mol, (cob,cvb,cob,cvb))
eri = eri.reshape(noccb,nvirb,noccb,nvirb)
e_denom = 1.0/(eob.reshape(-1, 1, 1, 1) - evb.reshape(-1, 1, 1) + \
               eob.reshape(-1, 1) - evb)
rdm2_ss_bb = numpy.zeros((noccb,nvirb,noccb,nvirb))
rdm2_ss_bb = einsum('iajb,iajb->iajb', eri, e_denom)
rdm2_ss_bb -= einsum('ibja,iajb->iajb', eri, e_denom)
echeck_bb = einsum('iajb,iajb->', eri, rdm2_ss_bb)*0.5
print('E(MP2 SS BB) is %12.8f' % echeck_bb)

# The alpha-beta part
eri = ao2mo.general(mol, (coa,cva,cob,cvb))
eri = eri.reshape(nocca,nvira,noccb,nvirb)
e_denom = 1.0/(eoa.reshape(-1, 1, 1, 1) - eva.reshape(-1, 1, 1) + \
               eob.reshape(-1, 1) - evb)
rdm2_os_ab = numpy.zeros((nocca,nvira,noccb,nvirb))
rdm2_os_ab = einsum('iajb,iajb->iajb', eri, e_denom)
echeck_ab = einsum('iajb,iajb->', eri, rdm2_os_ab)*0.5
print('E(MP2 OS AB) is %12.8f' % echeck_ab)

print('E(UMP2) = %12.8f' % (echeck_aa+echeck_bb+2*echeck_ab))
print('E(UHF+UMP2) = %12.8f' % (echeck_aa+echeck_bb+2*echeck_ab+ehf))

