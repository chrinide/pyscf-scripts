#!/usr/bin/env python

import numpy, scipy, time
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
nso = nocc + nvir
lib.logger.info(mf,"* Core orbitals: %d" % ncore)
lib.logger.info(mf,"* Virtual orbitals: %d" % nvir)

eri_ao = ao2mo.restore(1,mf._eri,nao0)
eri_ao = eri_ao.reshape((nao0,nao0,nao0,nao0))
def spin_block(eri):
    identity = numpy.eye(2)
    eri = numpy.kron(identity, eri)
    return numpy.kron(identity, eri.T)
t = time.time()
lib.logger.debug(mf,'Start building spin AO eri')
eri_ao = spin_block(eri_ao)
eri_ao = eri_ao - eri_ao.transpose(0,3,2,1)
hao = numpy.kron(numpy.eye(2), mf.get_hcore())
lib.logger.debug(mf,'.. finished in %.3f seconds.' % (time.time()-t))

o = slice(0, nocc)
v = slice(nocc, None)
x = numpy.newaxis

c = coeff[:,ncore:]
n = c.shape[1]
t = time.time()
lib.logger.debug(mf,'Start transform AO to MO basis')
eri_mo = ao2mo.general(eri_ao, (c,c,c,c), compact=False)
lib.logger.debug(mf,'.. finished in %.3f seconds.' % (time.time()-t))
eri_mo = eri_mo.reshape(n,n,n,n)
def ao_to_mo(hao, c):
    return c.T.dot(hao).dot(c)
hmo = ao_to_mo(hao, c)

# Intialize amplitudes and RDMs
t2 = numpy.zeros((nvir,nocc,nvir,nocc)) 
opdm_corr = numpy.zeros((nso,nso))
opdm_ref = numpy.zeros((nso,nso))
opdm_ref[o, o] = numpy.identity(nocc)
tpdm_corr = numpy.zeros((nso,nso,nso,nso))
# Initialize the rotation matrix parameter 
X = numpy.zeros((nso,nso))

e_nuc = mol.energy_nuc()
maxiter = 50
e_conv = 1.0e-8
e_omp2 = 0.0
e_old = 0.0 
for it in range(maxiter+1):
    e_old = e_omp2

    f = hmo + numpy.einsum('pqii->pq', eri_mo[:,:,o,o])
    fprime = f.copy()
    numpy.fill_diagonal(fprime, 0)
    eps = f.diagonal()

    t1 = eri_mo[v,o,v,o]
    t = time.time()
    lib.logger.debug(mf,'Start building amplitudes')
    t2 = numpy.einsum('ac,cibj->aibj', fprime[v,v], t2)
    t3 = numpy.einsum('ki,akbj->aibj', fprime[o,o], t2)
    lib.logger.debug(mf,'.. finished in %.3f seconds.' % (time.time()-t))
    t2 = t1 + t2 - t2.transpose((2,1,0,3)) \
            - t3 + t3.transpose((0,3,2,1))
    t2 /= (- eps[v,x,x,x] + eps[x,o,x,x] -
             eps[x,x,v,x] + eps[x,x,x,o])
   
    t = time.time()
    lib.logger.debug(mf,'Start building 1-RDM')
    opdm_corr[v,v] = 0.5*numpy.einsum('iajc,bicj->ba', t2.T, t2)
    opdm_corr[o,o] = -0.5*numpy.einsum('jakb,aibk->ji', t2.T, t2)
    lib.logger.debug(mf,'.. finished in %.3f seconds.' % (time.time()-t))
    opdm = opdm_corr + opdm_ref 

    tpdm_corr[v,o,v,o] = t2
    tpdm_corr[o,v,o,v] = t2.T
    t = time.time()
    lib.logger.debug(mf,'Start building 2-RDM')
    tpdm2 = numpy.einsum('rp,sq->rpsq', opdm_corr, opdm_ref)
    tpdm3 = numpy.einsum('rp,sq->rpsq', opdm_ref, opdm_ref)
    lib.logger.debug(mf,'.. finished in %.3f seconds.' % (time.time()-t))
    tpdm = tpdm_corr + \
           tpdm2 - tpdm2.transpose((2,1,0,3)) - \
           tpdm2.transpose((0,3,2,1)) + tpdm2.transpose((2,3,0,1)) + \
           tpdm3 - tpdm3.transpose((2,1,0,3))

    # Newton-Raphson step
    t = time.time()
    lib.logger.info(mf,'Start building NR')
    F = numpy.einsum('pr,rq->pq', hmo, opdm) + \
        0.5*numpy.einsum('psrt,sqtr->pq', eri_mo, tpdm)
    X[v,o] = ((F - F.T)[v,o])/(-eps[v,x] + eps[x,o])
    # Build Newton-Raphson orbital rotation matrix
    U = scipy.linalg.expm(X - X.T)
    # Rotate spin-orbital coefficients
    c = c.dot(U)
    lib.logger.info(mf,'.. finished in %.3f seconds.' % (time.time()-t))

    t = time.time()
    lib.logger.debug(mf,'Start rotating integrals')
    hmo = ao_to_mo(hao, c)
    eri_mo = ao2mo.general(eri_ao, (c,c,c,c))
    lib.logger.debug(mf,'.. finished in %.3f seconds.' % (time.time()-t))

    e_omp2 = e_nuc + numpy.einsum('pq,qp->', hmo, opdm) + \
             0.25*numpy.einsum('prqs,rpsq ->', eri_mo, tpdm)

    de = e_omp2 - e_old
    lib.logger.info(mf,'Iteration %3d: energy = %4.12f de = %1.5e' % (it, e_omp2, de))
    
    if abs(de) < e_conv:
        lib.logger.info(mf,"OMP2 Iterations have converged!")
        break
    if (it == maxiter):
        raise Exception("Maximum number of iterations exceeded.")

lib.logger.info(mf,"!*** E(HF+OMP2): %12.8f" % e_omp2)

