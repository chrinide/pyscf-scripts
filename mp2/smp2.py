#!/usr/bin/env python

import numpy
from pyscf import gto, scf, lib, ao2mo
from pyscf.tools import molden
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

mf = scf.density_fit(scf.RHF(mol))
mf.auxbasis = 'aug-cc-pvdz-jkfit'
mf.chkfile = name+'.chk'
mf.level_shift = 0.5
mf = scf.addons.remove_linear_dep_(mf)
mf = scf.newton(mf)
mf.kernel()
dm = mf.make_rdm1()
mf.level_shift = 0.0
ehf = mf.kernel(dm)

nao, nmo = mf.mo_coeff.shape
nocc = mol.nelectron/2
nvir = nmo - nocc
co = mf.mo_coeff[:,:nocc]
cv = mf.mo_coeff[:,nocc:]
eo = mf.mo_energy[:nocc]
ev = mf.mo_energy[nocc:]

naux = mf._cderi.shape[0]
dferi = numpy.empty((naux,nao,nao))
for i in range(naux):
    dferi[i] = lib.unpack_tril(mf._cderi[i])

eri_mo = numpy.einsum('rj,Qrs->Qjs', co, dferi)
eri_mo = numpy.einsum('sb,Qjs->Qjb', cv, eri_mo)
Qia = eri_mo
e_denom = 1.0/(eo.reshape(-1,1,1,1)-ev.reshape(-1,1,1)+eo.reshape(-1,1)-ev)

nsample = 500
e_srimp2 = 0.0

# Loop over samples to reduce stochastic noise
lib.logger.info(mf, "Starting sample loop...")
for x in range(nsample):
    # Create two random vector
    vec = numpy.random.choice([-1, 1], size=(Qia.shape[0]))
    vecp = numpy.random.choice([-1, 1], size=(Qia.shape[0]))
    # Generate first R matrices
    ia = einsum("Q,Qia->ia", vec, Qia)
    iap = einsum("Q,Qia->ia", vecp, Qia)
    # Caculate sRI-MP2 correlation energy
    e_srimp2 += 2.0*numpy.einsum('iajb,ia,ia,jb,jb->', e_denom, ia, iap, ia, iap)
    e_srimp2 -= numpy.einsum('iajb,ia,ib,jb,ja->', e_denom, ia, iap, ia, iap)

e_srimp2 /= float(nsample)

# Print sample energy to output
lib.logger.info(mf, "Number of samples:                 % 16d" % nsample)
lib.logger.info(mf, "sRI-MP2 correlation sample energy: % 16.10f" % e_srimp2)

