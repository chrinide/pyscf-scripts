#!/usr/bin/env python

import numpy
from pyscf import gto, scf, lib, ao2mo
einsum = lib.einsum

name = 'ch4'

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
mf.chkfile = name+'.chk'
mf.level_shift = 0.5
mf = scf.addons.remove_linear_dep_(mf)
mf = scf.newton(mf)
mf.kernel()
dm = mf.make_rdm1()
mf.level_shift = 0.0
ehf = mf.kernel(dm)
 
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
lib.logger.info(mf,"* Core orbitals: %d" % ncore)
lib.logger.info(mf,"* Virtual orbitals: %d" % (len(ev)))
eri_mo = ao2mo.general(mf._eri, (co,cv,co,cv), compact=False)
eri_mo = eri_mo.reshape(nocc,nvir,nocc,nvir)
e_denom = 1.0/(eo.reshape(-1,1,1,1)-ev.reshape(-1,1,1)+eo.reshape(-1,1)-ev)
t2 = numpy.zeros((nocc,nvir,nocc,nvir))

# erf(omega * r12) / r12
omega = 0.5
mol.set_range_coulomb(omega)
erf_r12 = mol.intor('cint2e_sph')
erf_r12 = ao2mo.restore(8,erf_r12,nao)
eri_mo_erfr12 = ao2mo.general(erf_r12, (co,cv,co,cv), compact=False)
eri_mo_erfr12 = eri_mo_erfr12.reshape(nocc,nvir,nocc,nvir)
erfc_r12 = eri_mo - eri_mo_erfr12
#erfc_r12 = 1.0 - eri_mo_erfr12

diff = numpy.linalg.norm((eri_mo_erfr12 + erfc_r12) - eri_mo)
lib.logger.info(mf,'*** Diff %s' % diff)

t2 = 2*einsum('iajb,iajb->iajb', eri_mo_erfr12, e_denom)
t2 -= einsum('ibja,iajb->iajb', eri_mo_erfr12, e_denom)
e_mp2_lr = numpy.einsum('iajb,iajb->', eri_mo_erfr12, t2, optimize=True)
lib.logger.info(mf, 'E(MP2-LR) is %12.8f' % e_mp2_lr)

t2 = 2*einsum('iajb,iajb->iajb', erfc_r12, e_denom)
t2 -= einsum('ibja,iajb->iajb', erfc_r12, e_denom)
e_mp2_sr = numpy.einsum('iajb,iajb->', erfc_r12, t2, optimize=True)
lib.logger.info(mf, 'E(MP2-SR) is %12.8f' % e_mp2_sr)

t2 = 2.0*einsum('iajb,iajb->iajb', eri_mo, e_denom)
t2 -= einsum('ibja,iajb->iajb', eri_mo, e_denom)
e_mp2 = numpy.einsum('iajb,iajb->', eri_mo, t2, optimize=True)
lib.logger.info(mf, "!*** E(MP2): %12.8f" % e_mp2)
lib.logger.info(mf, "!**** E(HF+MP2): %12.8f" % (e_mp2+ehf))

