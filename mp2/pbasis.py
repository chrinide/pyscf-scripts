#!/usr/bin/env python

import numpy, scipy, os, sys
from pyscf import gto, scf, ao2mo, lib
einsum = lib.einsum

mol = gto.Mole()
mol.basis = 'cc-pvdz'
mol.atom = '''
C     -0.662614     -0.000000     -0.000000
C      0.662614     -0.000000     -0.000000
F     -1.388214     -1.100388      0.000000
F      1.388214     -1.100388      0.000000
F     -1.388214      1.100388      0.000000
F      1.388214      1.100388      0.000000
'''
mol.verbose = 4
mol.spin = 0
mol.symmetry = 0
mol.charge = 0
mol.build()

mf = scf.RHF(mol)
ehf = mf.kernel()

ncore = 6
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
e_denom = 1.0/(eo.reshape(-1,1,1,1)-ev.reshape(-1,1,1)+eo.reshape(-1,1)-ev)
eri_mo = ao2mo.general(mf._eri, (co,cv,co,cv), compact=False)
eri_mo = eri_mo.reshape(nocc,nvir,nocc,nvir)
lib.logger.info(mf,"Virtual orbitals: %d" % (len(ev)))
rdm2_mp2 = numpy.zeros((nocc,nvir,nocc,nvir))
rdm2_mp2 = 2.0*einsum('iajb,iajb->iajb', eri_mo, e_denom)
rdm2_mp2 -= einsum('ibja,iajb->iajb', eri_mo, e_denom)
e_mp2 = numpy.einsum('iajb,iajb->', eri_mo, rdm2_mp2, optimize=True)
lib.logger.info(mf,"E(MP2): %12.8f" % e_mp2)
lib.logger.info(mf,"E(HF+MP2): %12.8f" % (e_mp2+ehf))

pmol = mol.copy()
pmol.atom = mol._atom
pmol.unit = 'B'
pmol.symmetry = False
pmol.basis = 'sto-6g'
pmol.build(False, False)

sb = mol.intor('int1e_ovlp')
sbinv = numpy.linalg.inv(sb)
sbs = gto.intor_cross('int1e_ovlp', mol, pmol)
ss = reduce(numpy.dot, (sbs.T,sbinv,sbs))
ssinv = numpy.linalg.inv(ss)
sslow = scipy.linalg.sqrtm(ss)
h = numpy.dot(mf.mo_coeff*mf.mo_energy, mf.mo_coeff.T)
hb = reduce(numpy.dot, (sb,h,sb))
hs = reduce(numpy.dot, (sbs.T,h,sbs))
mo_energy, mo_coeff = mf.eig(hs, ss)

# Composite basis big_prim x small_states
cp = reduce(numpy.dot, (sbinv, sbs, mo_coeff))
nocc = mol.nelectron//2
for i in range(nocc):
    cp[:,i] = mf.mo_coeff[:,i]
sp = reduce(numpy.dot, (cp.T, sb, cp))
hp = reduce(numpy.dot, (cp.T, hb, cp)) 
mo_energy, mo_coeff = mf.eig(hp, sp)
cpp = numpy.dot(cp,mo_coeff) 

ncore = 6
nao, nmo = cpp.shape
nocc = mol.nelectron//2 - ncore
nvir = nmo - nocc - ncore
mo_core = cpp[:,:ncore]
mo_occ = cpp[:,ncore:ncore+nocc]
mo_vir = cpp[:,ncore+nocc:]
co = mo_occ
cv = mo_vir
eo = mo_energy[ncore:ncore+nocc]
ev = mo_energy[ncore+nocc:]
e_denom = 1.0/(eo.reshape(-1,1,1,1)-ev.reshape(-1,1,1)+eo.reshape(-1,1)-ev)
eri_mo = ao2mo.general(mf._eri, (co,cv,co,cv), compact=False)
eri_mo = eri_mo.reshape(nocc,nvir,nocc,nvir)
lib.logger.info(mf,"Virtual orbitals: %d" % (len(ev)))
rdm2_mp2 = numpy.zeros((nocc,nvir,nocc,nvir))
rdm2_mp2 = 2.0*einsum('iajb,iajb->iajb', eri_mo, e_denom)
rdm2_mp2 -= einsum('ibja,iajb->iajb', eri_mo, e_denom)
e_mp2 = numpy.einsum('iajb,iajb->', eri_mo, rdm2_mp2, optimize=True)
lib.logger.info(mf,"E(MP2): %12.8f" % e_mp2)
lib.logger.info(mf,"E(HF+MP2): %12.8f" % (e_mp2+ehf))

