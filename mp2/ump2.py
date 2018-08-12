#!/usr/bin/env python

import numpy
from functools import reduce
from pyscf import scf, gto, mp, lib, ao2mo
from pyscf.tools import molden
einsum = lib.einsum

name = 'ump2'

mol = gto.Mole()
mol.atom = [['O', (0.,   0., 0.)],
            ['O', (1.21, 0., 0.)]]
mol.basis = 'sto-3g'
mol.spin = 2
mol.verbose = 4
mol.build()

mf = scf.UHF(mol)
ehf = mf.kernel()
nao,nmo = mf.mo_coeff[0].shape

frozen = [[0,1],[0,1]]
pt2 = mp.UMP2(mf, frozen=frozen)
emp2, t2 = pt2.kernel()

rdm1a, rdm1b = pt2.make_rdm1()
rdm2aa, rdm2ab, rdm2bb = pt2.make_rdm2()
rdm2ba = rdm2ab.transpose(2,3,0,1)

occ_a = mf.mo_occ[0]
occ_b = mf.mo_occ[1]
mo_a = mf.mo_coeff[0]
mo_b = mf.mo_coeff[1]
nmoa = mo_a.shape[1]
nmob = mo_b.shape[1]

coeff = numpy.hstack([mo_a,mo_b])
occ = numpy.hstack([occ_a,occ_b])
with open(name+'.mol', 'w') as f2:
    molden.header(mol, f2)
    molden.orbital_coeff(mol, f2, coeff, occ=occ)

eriaa = ao2mo.kernel(mf._eri, mo_a, compact=False).reshape([nmoa]*4)
eribb = ao2mo.kernel(mf._eri, mo_b, compact=False).reshape([nmob]*4)
eriab = ao2mo.kernel(mf._eri, (mo_a,mo_a,mo_b,mo_b), compact=False)
eriab = eriab.reshape([nmoa,nmoa,nmob,nmob])
hcore = mf.get_hcore()

h1a = reduce(numpy.dot, (mo_a.T.conj(), hcore, mo_a))
h1b = reduce(numpy.dot, (mo_b.T.conj(), hcore, mo_b))
e1 = einsum('ij,ji', h1a, rdm1a)
e1 += einsum('ij,ji', h1b, rdm1b)
e1 += einsum('ijkl,ijkl', eriaa, rdm2aa)*0.5
e1 += einsum('ijkl,ijkl', eriab, rdm2ab)
e1 += einsum('ijkl,ijkl', eribb, rdm2bb)*0.5
e1 += mol.energy_nuc()
lib.logger.info(pt2,"!**** 1/2-RDM energy: %12.8f" % (e1))

den_file = name + '.den'
fspt = open(den_file,'w')
fspt.write('CCIQA\n')
fspt.write('1-RDM:\n')
for i in range(nmo):
    for j in range(nmo):
        fspt.write('%i %i %.10f\n' % ((i+1), (j+1), rdm1a[i,j]))
        fspt.write('%i %i %.10f\n' % ((i+1+nmo), (j+1+nmo), rdm1b[i,j]))
fspt.write('2-RDM:\n')
for i in range(nmo):
    for j in range(nmo):
        for k in range(nmo):
            for l in range(nmo):
                if (abs(rdm2aa[i,j,k,l]) > 1e-8):
                    fspt.write('%i %i %i %i %.10f\n' % ((i+1), \
                    (j+1), (k+1), (l+1), rdm2aa[i,j,k,l]))
                if (abs(rdm2bb[i,j,k,l]) > 1e-8):
                    fspt.write('%i %i %i %i %.10f\n' % ((i+1+nmo), \
                    (j+1+nmo), (k+1+nmo), (l+1+nmo), rdm2bb[i,j,k,l]))
                if (abs(rdm2ab[i,j,k,l]) > 1e-8):
                    fspt.write('%i %i %i %i %.10f\n' % ((i+1), \
                    (j+1), (k+1+nmo), (l+1+nmo), rdm2ab[i,j,k,l]))
                if (abs(rdm2ba[i,j,k,l]) > 1e-8):
                    fspt.write('%i %i %i %i %.10f\n' % ((i+1+nmo), \
                    (j+1+nmo), (k+1), (l+1), rdm2ba[i,j,k,l]))
fspt.close()                    
