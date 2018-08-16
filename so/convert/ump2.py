#!/usr/bin/env python

import numpy
from pyscf import gto, scf, lib, ao2mo
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
mol.charge = 0
mol.spin = 0
mol.symmetry = 1
mol.verbose = 4
mol.build()

mf = scf.UHF(mol)
ehf = mf.kernel()
nao0, nmo0 = mf.mo_coeff[0].shape

ncore = (1,1)
nelec = mol.nelectron
nelecb = nelec//2
neleca = nelec - nelecb
ncorea = ncore[0]
ncoreb = ncore[1]
nocca = neleca - ncorea
nvira = nmo0 - nocca - ncorea
noccb = nelecb - ncoreb
nvirb = nmo0 - noccb - ncoreb
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
co = numpy.block([[coa, numpy.zeros_like(cob)],
                  [numpy.zeros_like(coa), cob]])
cv = numpy.block([[cva, numpy.zeros_like(cvb)],
                  [numpy.zeros_like(cva), cvb]])
eo = numpy.hstack((eoa,eob)) 
ev = numpy.hstack((eva,evb))                 
e_denom = 1.0/(eo.reshape(-1,1,1,1)-ev.reshape(-1,1,1)+eo.reshape(-1,1)-ev)

eri_ao = ao2mo.restore(1,mf._eri,nao0)
eri_ao = eri_ao.reshape((nao0,nao0,nao0,nao0))
def spin_block(eri):
    identity = numpy.eye(2)
    eri = numpy.kron(identity, eri)
    return numpy.kron(identity, eri.T)
eri_ao = spin_block(eri_ao)
eri_mo = ao2mo.general(eri_ao, (co,cv,co,cv), compact=False)
eri_mo = eri_mo.reshape(nocc,nvir,nocc,nvir)

# Antisymmetrize:
# <pr||qs> = <pr|qs> - <ps|qr>
#eri_mo = eri_mo - eri_mo.transpose(0,3,2,1)
#t2 = numpy.zeros((nocc,nvir,nocc,nvir))
#t2 = einsum('iajb,iajb->iajb', eri_mo, e_denom)
#e_mp2 = 0.25*numpy.einsum('iajb,iajb->', eri_mo, t2, optimize=True)
#lib.logger.info(mf,"!*** E(MP2): %12.8f" % e_mp2)
#lib.logger.info(mf,"!**** E(HF+MP2): %12.8f" % (e_mp2+ehf))

t2 = numpy.zeros((nocc,nvir,nocc,nvir))
t2 = numpy.einsum('iajb,iajb->iajb', \
     eri_mo - eri_mo.transpose(0,3,2,1), e_denom, optimize=True)
e_mp2 = 0.5*numpy.einsum('iajb,iajb->', eri_mo, t2, optimize=True)
lib.logger.info(mf,"!*** E(MP2): %12.8f" % e_mp2)
lib.logger.info(mf,"!**** E(HF+MP2): %12.8f" % (e_mp2+ehf))

