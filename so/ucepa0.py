#!/usr/bin/env python

import numpy
from pyscf import gto, scf, lib, ao2mo
from pyscf.tools import molden
einsum = lib.einsum

mol = gto.Mole()
mol.basis = '6-31g'
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

c = coeff[:,ncore:]
n = c.shape[1]

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
v = slice(nocc, None)
n = numpy.newaxis
eri_mo = eri_mo - eri_mo.transpose(0,3,2,1)

t2 = numpy.zeros((nvir,nocc,nvir,nocc))
eo = energy[ncore:ncore+nocc]
ev = energy[ncore+nocc:]
e_denom = 1.0/(-ev.reshape(-1,1,1,1)+eo.reshape(-1,1,1)-ev.reshape(-1,1)+eo)

e_cepa0 = 0.0
maxiter = 50
mp2 = eri_mo[v, o, v, o]
mp2 = mp2*e_denom
for it in range(maxiter+1):
    e_old = e_cepa0
    cepa1 = 0.5*numpy.einsum('acbd,cidj->aibj', eri_mo[v,v,v,v], t2)
    cepa2 = 0.5*numpy.einsum('kilj,akbl->aibj', eri_mo[o,o,o,o], t2)
    cepa3a = numpy.einsum('aikc,bjck->aibj', eri_mo[v,o,o,v], t2)
    cepa3b = -cepa3a.transpose(2,1,0,3)
    cepa3c = -cepa3a.transpose(0,3,2,1)
    cepa3d = cepa3a.transpose(2,3,0,1)
    cepa3 = cepa3a + cepa3b + cepa3c + cepa3d
    t2_new = e_denom*(cepa1 + cepa2 + cepa3) + mp2
    e_cepa0 = 0.25*numpy.einsum('iajb,aibj->', eri_mo[o,v,o,v], t2_new)
    t2 = t2_new
    de = e_cepa0 - e_old
    lib.logger.info(mf,'Iteration %3d: energy = %4.12f de = %1.5e' % (it, e_cepa0, de))
    if abs(de) < 1.e-8:
        lib.logger.info(mf,"CEPA0 Iterations have converged!")
        break
    if (it == maxiter):
        raise Exception("Maximum number of iterations exceeded.")

lib.logger.info(mf,'CEPA0 Correlation Energy: %5.15f' % (e_cepa0))
lib.logger.info(mf,'CEPA0 Total Energy: %5.15f' % (e_cepa0 + ehf))

