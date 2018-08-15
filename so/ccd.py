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
eri_mo = eri_mo.transpose(0,2,1,3)

t_amp = numpy.zeros((nvir,nvir,nocc,nocc))
e_abij = 1.0/(-energy[v,n,n,n] - energy[n,v,n,n] + energy[n,n,o,n] + energy[n,n,n,o]) 

maxiter = 50
e_ccd = 0.0
for cc_iter in range(1,maxiter+1):
    e_old = e_ccd

    mp2 = eri_mo[v,v,o,o]
    cepa1 = 0.5*numpy.einsum('abcd,cdij->abij', eri_mo[v,v,v,v], t_amp)
    cepa2 = 0.5*numpy.einsum('klij,abkl->abij', eri_mo[o,o,o,o], t_amp)
    cepa3a = numpy.einsum('akic,bcjk->abij', eri_mo[v,o,o,v], t_amp)
    cepa3b = -cepa3a.transpose(1,0,2,3)
    cepa3c = -cepa3a.transpose(0,1,3,2)
    cepa3d = cepa3a.transpose(1,0,3,2)
    cepa3 = cepa3a + cepa3b + cepa3c + cepa3d

    ccd1a_tmp = numpy.einsum('klcd,bdkl->cb', eri_mo[o,o,v,v], t_amp)
    ccd1a = numpy.einsum("cb,acij->abij", ccd1a_tmp, t_amp)
    ccd1b  = -ccd1a.transpose(1,0,2,3)
    ccd1   = -0.5*(ccd1a + ccd1b)
    ccd2a_tmp = numpy.einsum('klcd,cdjl->jk', eri_mo[o,o,v,v], t_amp)
    ccd2a = numpy.einsum("jk,abik->abij", ccd2a_tmp, t_amp)
    ccd2b = -ccd2a.transpose(0,1,3,2)
    ccd2 = -0.5 * (ccd2a + ccd2b)
    ccd3_tmp = numpy.einsum("klcd,cdij->klij", eri_mo[o,o,v,v], t_amp)
    ccd3 = 0.25*numpy.einsum("klij,abkl->abij", ccd3_tmp, t_amp)
    ccd4a_tmp = numpy.einsum("klcd,acik->laid", eri_mo[o,o,v,v], t_amp)
    ccd4a = numpy.einsum("laid,bdjl->abij", ccd4a_tmp, t_amp)
    ccd4b = -ccd4a.transpose(0,1,3,2)
    ccd4 = (ccd4a + ccd4b)

    t_amp_new = e_abij*(mp2 + cepa1 + cepa2 + cepa3 + ccd1 + ccd2 + ccd3 + ccd4)
    e_ccd = 0.25*numpy.einsum('ijab,abij->', eri_mo[o,o,v,v], t_amp_new)
    t_amp = t_amp_new
    de = e_ccd - e_old
    lib.logger.info(mf,'Iteration %3d: energy = %4.12f de = %1.5e' % (cc_iter, e_ccd, de))
    if abs(de) < 1.e-8:
        lib.logger.info(mf,"CCD Iterations have converged!")
        break
    if (cc_iter == maxiter):
        raise Exception("Maximum number of iterations exceeded.")

lib.logger.info(mf,'CCD Correlation Energy: %5.15f' % (e_ccd))
lib.logger.info(mf,'CCD Total Energy: %5.15f' % (e_ccd + ehf))

