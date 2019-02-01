#!/usr/bin/env python

import numpy
from pyscf import gto, scf, ao2mo, fci, mcscf
from pyscf.tools import wfn_format

mol = gto.Mole()
mol.basis = 'sto-6g'
mol.atom = '''
H  0.0000  0.0000  0.0000
H  0.0000  0.0000  0.7500
    '''
mol.verbose = 4
mol.spin = 0
mol.charge = 0
mol.symmetry = 1
mol.build()

mf = scf.RHF(mol)
mf.kernel()

nao, nmo = mf.mo_coeff.shape
eri_mo = ao2mo.kernel(mf._eri, mf.mo_coeff[:,:nmo], compact=False)
eri_mo = eri_mo.reshape(nmo,nmo,nmo,nmo)
h1 = reduce(numpy.dot, (mf.mo_coeff[:,:nmo].T, mf.get_hcore(), mf.mo_coeff[:,:nmo]))
print h1
print eri_mo

# GVB-PP
ndets = 2
h = numpy.zeros((ndets,ndets))
for i in range(ndets):
    h[i,i] = 2.0*h1[i,i] + eri_mo[i,i,i,i]
    for j in range(i):
        h[i,j] = eri_mo[i,j,i,j]
        h[j,i] = h[i,j]

e,c = numpy.linalg.eig(h)
e += mf.energy_nuc()
print('E(GVB-PP) = %s' % e)

cisolver = fci.FCI(mol, mf.mo_coeff)
cisolver.nroots = 10000
print('E(FCI) = %s' % cisolver.kernel()[0])

