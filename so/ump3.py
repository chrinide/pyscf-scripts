#!/usr/bin/env python

import numpy
from pyscf import gto, scf, lib, ao2mo
from pyscf.tools import molden
einsum = lib.einsum

mol = gto.Mole()
mol.basis = 'cc-pvdz'
mol.atom = '''
O
H 1 1.1
H 1 1.1 2 104
'''
mol.charge = 0
mol.spin = 0
mol.symmetry = 1
mol.verbose = 4
mol.build()

mf = scf.UHF(mol)
ehf = mf.kernel()
nao0, nmo0 = mf.mo_coeff[0].shape

ca = mf.mo_coeff[0]
cb = mf.mo_coeff[1]
coeff = numpy.block([[ca            ,numpy.zeros_like(cb)],
                     [numpy.zeros_like(ca),            cb]])
occ = numpy.hstack((mf.mo_occ[0],mf.mo_occ[1]))
energy = numpy.hstack((mf.mo_energy[0],mf.mo_energy[1]))

idx = energy.argsort()
coeff = coeff[:,idx]
occ = occ[idx]
energy = energy[idx]

nao, nmo = coeff.shape
ncore = 0
nocc = mol.nelectron - ncore
nvir = nmo - nocc - ncore
lib.logger.info(mf,"* Core orbitals: %d" % ncore)
lib.logger.info(mf,"* Virtual orbitals: %d" % nvir)

mo_core = coeff[:,:ncore]
c = coeff[:,ncore:]
n = c.shape[1]
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
eri_mo = ao2mo.general(eri_ao, (c,c,c,c), compact=False)
eri_mo = eri_mo.reshape(n,n,n,n)
o = slice(0, nocc)
v = slice(nocc, eri_mo.shape[0])

t2 = numpy.zeros((nocc,nvir,nocc,nvir))
t2 = numpy.einsum('iajb,iajb->iajb', \
     eri_mo[o,v,o,v] - eri_mo[o,v,o,v].transpose(0,3,2,1), e_denom, optimize=True)
e_mp2 = 0.5*numpy.einsum('iajb,iajb->', eri_mo[o,v,o,v], t2, optimize=True)
lib.logger.info(mf,"!*** E(MP2): %12.8f" % e_mp2)
lib.logger.info(mf,"!**** E(HF+MP2): %12.8f" % (e_mp2+ehf))

# MP3 Correlation: [Szabo:1996] pp. 353, Eqn. 6.75
eri_mo = eri_mo - eri_mo.transpose(0,3,2,1)
eqn1 = 0.125*numpy.einsum('arbs,cadb,rcsd,crds->', t2, eri_mo[o,o,o,o], eri_mo[v,o,v,o], e_denom)
eqn2 = 0.125*numpy.einsum('arbs,rtsu,taub,atbu->', t2, eri_mo[v,v,v,v], eri_mo[v,o,v,o], e_denom)
eqn3 = numpy.einsum('arbs,ctsb,ratc,arct->', t2, eri_mo[o,v,v,o], eri_mo[v,o,v,o], e_denom)

e_mp3 = eqn1 + eqn2 + eqn3
lib.logger.info(mf,"!*** E(MP3): %12.8f" % e_mp3)
lib.logger.info(mf,"!*** E(MP2+MP3): %12.8f" % (e_mp2+e_mp3))
lib.logger.info(mf,"!**** E(HF+MP2+MP3): %12.8f" % (e_mp3+e_mp2+ehf))

