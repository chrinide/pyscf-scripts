#!/usr/bin/env python

import numpy
from pyscf import gto, scf, ao2mo, lib

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
mf.scf()

ncorea = 1
ncoreb = 1
nocc = mol.nelectron - ncorea -ncoreb
lib.logger.info(mf,"* Core orbitals: %d" % (ncorea+ncoreb))

ca = mf.mo_coeff[0][:,ncorea:]
cb = mf.mo_coeff[1][:,ncoreb:]
mo_coeff = numpy.array((ca,cb))
ea = mf.mo_energy[0][ncorea:]
eb = mf.mo_energy[1][ncoreb:]
mo_energy = numpy.array((ea,eb))
oa = mf.mo_occ[0][ncorea:]
ob = mf.mo_occ[1][ncoreb:]
mo_occ = numpy.array((oa,ob))

coa = mo_coeff[0][:,mo_occ[0]==1]
cva = mo_coeff[0][:,mo_occ[0]==0]
cob = mo_coeff[1][:,mo_occ[1]==1]
cvb = mo_coeff[1][:,mo_occ[1]==0]

eoa = mo_energy[0][mo_occ[0]==1]
eva = mo_energy[0][mo_occ[0]==0]
eob = mo_energy[1][mo_occ[1]==1]
evb = mo_energy[1][mo_occ[1]==0]

co = numpy.hstack((coa,cob))
cv = numpy.hstack((cva,cvb))
eo = numpy.hstack((eoa,eob))
ev = numpy.hstack((eva,evb))

no = co.shape[1]
nv = cv.shape[1]
noa = sum(mo_occ[0]==1)
nva = sum(mo_occ[0]==0)
nob = sum(mo_occ[1]==1)
nvb = sum(mo_occ[1]==0)

eri = ao2mo.general(mf._eri, (co,cv,co,cv))
eri = eri.reshape(no,nv,no,nv)
eri[:noa,nva:] = eri[noa:,:nva] = 0.0
eri[:,:,:noa,nva:] = eri[:,:,noa:,:nva] = 0.0

eov = eo.reshape(-1,1) - ev.reshape(-1)
e = 1.0/(eov.reshape(-1,1) + eov.reshape(-1))
e = e.reshape(eri.shape)
t2 = numpy.einsum('iajb,iajb->iajb', eri - eri.transpose(0,3,2,1), e)
emp2 = 0.5*numpy.einsum('iajb,iajb->', eri, t2)

print('E(UMP2) = %.9g' % emp2)

