#!/usr/bin/env python

import numpy, h5py
from pyscf import gto, scf, lib
from pyscf.tools.dump_mat import dump_tri

name = 'h2'
mol = lib.chkfile.load_mol(name+'.chk')
mo_coeff = scf.chkfile.load(name+'.chk', 'scf/mo_coeff')
mo_occ = scf.chkfile.load(name+'.chk', 'scf/mo_occ')
dm = scf.hf.make_rdm1(mo_coeff, mo_occ)
s = mol.intor('int1e_ovlp')

pop = numpy.einsum('ij,ji->',s,dm)
lib.logger.info(mol,'* Info on AO basis')
lib.logger.info(mol,'Population : %12.6f' % pop)

nmo = mo_coeff.shape[1]
coeff = mo_coeff[:,:nmo]
rdm1 = numpy.zeros((nmo,nmo))
for i in range(mol.nelectron//2):
    rdm1[i,i] = 2.0

# from AO basis to MO basis
s = coeff.T.dot(s).dot(coeff)
pop = numpy.einsum('ij,ji->',s,rdm1)
lib.logger.info(mol,'* Info on MO basis')
lib.logger.info(mol,'Population : %12.6f' % pop)

# From MO basis to AO basis
coeff = numpy.linalg.inv(coeff)
s = coeff.T.dot(s).dot(coeff)
pop = numpy.einsum('ij,ji->',s,dm)
lib.logger.info(mol,'* Backtransform from MO to AO basis')
lib.logger.info(mol,'Population : %12.6f' % pop)
lib.logger.info(mol,'* Overlap on AO basis')
dump_tri(mol.stdout, s, ncol=15, digits=5, start=0)

# Reference AO basis
with h5py.File(name+'_integrals.h5') as f:
    sref = f['molecule/overlap'].value
lib.logger.info(mol,'* REF Overlap on AO basis')
dump_tri(mol.stdout, sref, ncol=15, digits=5, start=0)

# Read info from aom program
aom = numpy.zeros((mol.natm,nmo,nmo))
totaom = numpy.zeros((nmo,nmo))
with h5py.File(name+'.chk.h5') as f:
    for i in range(mol.natm):
        idx = 'ovlp'+str(i)
        aom[i] = f[idx+'/aom'].value
for i in range(mol.natm):
    lib.logger.info(mol,'Follow AOM for atom %d' % i)
    dump_tri(mol.stdout, aom[i], ncol=15, digits=5, start=0)
    totaom += aom[i]
lib.logger.info(mol,'Follow total AOM')
dump_tri(mol.stdout, totaom, ncol=15, digits=5, start=0)

# Transform for example AOM from atom 0 on MO basis to AO basis
lib.logger.info(mol,'* AOM for atom 0 on AO basis')
aom0 = coeff.T.dot(aom[0]).dot(coeff)
dump_tri(mol.stdout, aom0, ncol=15, digits=5, start=0)
pop = numpy.einsum('ij,ji->',aom0,dm)
lib.logger.info(mol,'Population : %12.6f' % pop)

# Transform total MO aom to AO basis and check diff
lib.logger.info(mol,'* Total AOM on AO basis')
totaom = coeff.T.dot(totaom).dot(coeff)
dump_tri(mol.stdout, totaom, ncol=15, digits=5, start=0)
diff = numpy.linalg.norm(totaom-sref)
lib.logger.info(mol,'* Diff in AOM on %f' % diff)

