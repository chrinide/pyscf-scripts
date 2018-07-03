#!/usr/bin/env python

import numpy
from pyscf import gto, scf, lib, ao2mo
from pyscf.tools import molden
einsum = lib.einsum

mol = gto.Mole()
mol.basis = 'aug-cc-pvdz'
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
 
ncore = 1
 
nao, nmo = mf.mo_coeff.shape
nocc = mol.nelectron//2 - ncore
nvir = nmo - nocc - ncore
mo_core = mf.mo_coeff[:,:ncore]
mo_occ = mf.mo_coeff[:,ncore:ncore+nocc]
mo_vir = mf.mo_coeff[:,ncore+nocc:]
co = mo_occ
cv = mo_vir
eo = mf.mo_energy[ncore:ncore+nocc]
ev = mf.mo_energy[ncore+nocc:]
eri_mo = ao2mo.general(mf._eri, (co,cv,co,cv), compact=False)
eri_mo = eri_mo.reshape(nocc,nvir,nocc,nvir)
lib.logger.info(mf,"* Core orbitals: %d" % ncore)
lib.logger.info(mf,"* Virtual orbitals: %d" % (len(ev)))
 
e_denom = 1.0/(eo.reshape(-1,1,1,1)-ev.reshape(-1,1,1)+eo.reshape(-1,1)-ev)
t2_os = numpy.einsum('iajb,iajb->iajb', eri_mo-eri_mo.swapaxes(1,3), e_denom) 
t2_ss = numpy.einsum('iajb,iajb->iajb', eri_mo, e_denom)  
MP2corr_SS = numpy.einsum('iajb,iajb->', t2_os, eri_mo, optimize=True) 
MP2corr_OS = numpy.einsum('iajb,iajb->', t2_ss, eri_mo, optimize=True)  
MP2corr_E = MP2corr_SS + MP2corr_OS
MP2_E = ehf + MP2corr_E
SCS_MP2corr_E = MP2corr_SS/3.0 + MP2corr_OS*(6.0/5.0)
SCS_MP2_E = ehf + SCS_MP2corr_E
lib.logger.info(mf, 'MP2 SS correlation energy: %12.8f' % MP2corr_SS)
lib.logger.info(mf, 'MP2 OS correlation energy: %12.8f' % MP2corr_OS)
lib.logger.info(mf, 'MP2 correlation energy: %12.8f' % MP2corr_E)
lib.logger.info(mf, 'MP2 total energy: %12.8f' % MP2_E)
lib.logger.info(mf, 'SCS-MP2 correlation energy: %12.8f' % SCS_MP2corr_E)
lib.logger.info(mf, 'SCS-MP2 total energy: %12.8f' % SCS_MP2_E)

