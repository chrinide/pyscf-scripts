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

Ca = mf.mo_coeff[0]
Cb = mf.mo_coeff[1]
coeff = numpy.block([[Ca            ,numpy.zeros_like(Cb)],
                      [numpy.zeros_like(Ca),          Cb]])
occ = numpy.hstack((mf.mo_occ[0],mf.mo_occ[1]))
energy = numpy.hstack((mf.mo_energy[0],mf.mo_energy[1]))

idx = energy.argsort()
coeff = coeff[:,idx]
occ = occ[idx]
energy = energy[idx]

nao, nmo = coeff.shape
ncore = 2
nocc = mol.nelectron - ncore
nvir = nmo - nocc - ncore
lib.logger.info(mf,"* Core orbitals: %d" % ncore)
lib.logger.info(mf,"* Virtual orbitals: %d" % nvir)

mo_core = coeff[:,:ncore]
mo_occ = coeff[:,ncore:ncore+nocc]
mo_vir = coeff[:,ncore+nocc:]
co = mo_occ
cv = mo_vir
eo = energy[ncore:ncore+nocc]
ev = energy[ncore+nocc:]
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
eri_mo = eri_mo - eri_mo.transpose(0,3,2,1)
t2 = numpy.zeros((nocc,nvir,nocc,nvir))
t2 = einsum('iajb,iajb->iajb', eri_mo, e_denom)
e_mp2 = 0.25*numpy.einsum('iajb,iajb->', eri_mo, t2, optimize=True)
lib.logger.info(mf,"!*** E(MP2): %12.8f" % e_mp2)
lib.logger.info(mf,"!**** E(HF+MP2): %12.8f" % (e_mp2+ehf))

