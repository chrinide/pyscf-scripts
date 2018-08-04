#!/usr/bin/env python    

import numpy
from functools import reduce
from pyscf import gto, scf, lib, ao2mo, dft, tddft, gw
einsum = lib.einsum

mol = gto.Mole()
mol.basis = 'aug-cc-pvtz'
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

mf = dft.RKS(mol)
mf.xc = 'hf'
ehf = mf.kernel()

nocc = mol.nelectron//2
nmo = mf.mo_energy.size
nvir = nmo-nocc

td = tddft.dRPA(mf)
#td = tddft.RPA(mf)
td.verbose = 0
td.nstates = nocc*nvir
td.kernel()

gw = gw.GW(mf, td)
gw.linearized = False
gw.kernel()

nao, nmo = mf.mo_coeff.shape
ncore = 1
nocc = mol.nelectron//2 - ncore
nvir = nmo - nocc - ncore
lib.logger.info(mf,"* Core orbitals: %d" % ncore)
lib.logger.info(mf,"* Ocuppied orbitals: %d" % nocc)
lib.logger.info(mf,"* Virtual orbitals: %d" % nvir)

mo_core = mf.mo_coeff[:,:ncore]
mo_occ = mf.mo_coeff[:,ncore:ncore+nocc]
mo_vir = mf.mo_coeff[:,ncore+nocc:]
co = mo_occ
cv = mo_vir
eo = gw.mo_energy[ncore:ncore+nocc]
ev = gw.mo_energy[ncore+nocc:]

eri_mo = ao2mo.general(mf._eri, (co,cv,co,cv), compact=False)
eri_mo = eri_mo.reshape(nocc,nvir,nocc,nvir)
e_denom = 1.0/(eo.reshape(-1,1,1,1)-ev.reshape(-1,1,1)+eo.reshape(-1,1)-ev)
t2 = numpy.zeros((nocc,nvir,nocc,nvir))
t2 = 2.0*einsum('iajb,iajb->iajb', eri_mo, e_denom)
t2 -= einsum('ibja,iajb->iajb', eri_mo, e_denom)
e_mp2 = numpy.einsum('iajb,iajb->', eri_mo, t2, optimize=True)
lib.logger.info(mf,"!*** E(MP2): %12.8f" % e_mp2)
lib.logger.info(mf,"!**** E(HF+MP2): %12.8f" % (e_mp2+ehf))
