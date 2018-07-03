#!/usr/bin/env python

# TODO: core orbitals

import numpy
from pyscf import gto, scf, lib, ao2mo, symm
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
mol.symmetry = 'c1'
mol.verbose = 4
mol.build()

mf = scf.RHF(mol)
mf.chkfile = name+'.chk'
ehf = mf.kernel()
 
ncore = 0
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

lib.logger.info(mf,"* Building amplitudes")
e_denom = 1.0/(eo.reshape(-1,1,1,1)-ev.reshape(-1,1,1)+eo.reshape(-1,1)-ev)
t2amp = eri_mo * e_denom

# Compute occupied and virtual MP2 densities
rdm1 = numpy.zeros((nmo,nmo))
dij = 2.0*numpy.einsum('iajb,kajb->ik', t2amp, t2amp) - \
      1.0*numpy.einsum('iajb,kbja->ik', t2amp, t2amp)
dab = 2.0*numpy.einsum('iajb,icjb->ac', t2amp, t2amp) - \
      1.0*numpy.einsum('iajb,ibjc->ac', t2amp, t2amp)    
rdm1[:nocc, :nocc] -= 2.00 * dij
rdm1[nocc:, nocc:] += 2.00 * dab
for i in range(nocc):
    rdm1[i,i] += 2.0

orbsym = symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb, mf.mo_coeff[:,:nmo])
natocc, natorb = symm.eigh(-rdm1, orbsym)
#natocc, natorb = numpy.linalg.eigh(-rdm1)
for i, k in enumerate(numpy.argmax(abs(natorb), axis=0)):
    if natorb[k,i] < 0:
        natorb[:,i] *= -1
natorb = numpy.dot(mf.mo_coeff[:,:nmo], natorb)
natocc = -natocc
lib.logger.info(mf,"* Natural occupancies")
lib.logger.info(mf,"%s" % natocc)
lib.logger.info(mf,"* The sum of the natural occupation numbers is %6.4f" % numpy.sum(natocc))

thresh = 0.010
active = (thresh <= natocc) & (natocc <= 2-thresh)
lib.logger.info(mf,"* Threshold for active orbitals is %6.4f" % thresh)
lib.logger.info(mf,'Active Natural orbitals:')
for i in range(nmo):
    lib.logger.info(mf, 'orb: %d %s %6.4f' %(i,active[i],natocc[i]))
    actIndices = numpy.where(active)[0]
lib.logger.info(mf, 'Num active orbitals %d' % len(actIndices))
lib.logger.info(mf, 'active orbital indices %s' % actIndices)

# Compute second order density matrix and correlation energy
c = mf.mo_coeff
rdm2 = numpy.zeros((nmo,nmo,nmo,nmo))
rdm2[:nocc,nocc:,:nocc,nocc:] = t2amp*2.0 - t2amp.transpose(0,3,2,1)
rdm2[nocc:,:nocc,nocc:,:nocc] = rdm2[:nocc,nocc:,:nocc,nocc:].T
eri_mo = ao2mo.general(mf._eri, (c,c,c,c), compact=False)
eri_mo = eri_mo.reshape(nmo,nmo,nmo,nmo)
e2 = numpy.einsum('iajb,iajb->', rdm2, eri_mo)*0.5
lib.logger.info(mf,"Correlation energy is %12.8f" % e2)

