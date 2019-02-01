#!/usr/bin/env python

import numpy
from pyscf import gto, scf, ao2mo, fci, mcscf, x2c

mol = gto.Mole()
mol.basis = 'sto-6g'
mol.atom = '''
H  0.0000  0.0000  0.0000
H  0.0000  0.0000  0.7500
    '''
mol.verbose = 4
mol.spin = 0
mol.charge = 0
mol.symmetry = 0
mol.build()

mf = x2c.RHF(mol)
dm = mf.get_init_guess() + 0.1j
mf.kernel(dm)

nao, nmo = mf.mo_coeff.shape
eri_mo = ao2mo.kernel(mol, mf.mo_coeff[:,:nmo], compact=False, intor='int2e_spinor')
eri_mo = eri_mo.reshape(nmo,nmo,nmo,nmo)
h1 = reduce(numpy.dot, (mf.mo_coeff[:,:nmo].conj().T, mf.get_hcore(), mf.mo_coeff[:,:nmo]))

# GVB-PP
ndets = 4
h = numpy.zeros((ndets,ndets), dtype=numpy.complex128)
for i in range(ndets):
    h[i,i] = 2.0*h1[i,i] + eri_mo[i,i,i,i]
    for j in range(i):
        h[i,j] = eri_mo[i,j,i,j]
        h[j,i] = h[i,j]

e,c = numpy.linalg.eigh(h)
e += mf.energy_nuc()
print('E(GVB-PP) = %s' % e)

#cisolver = fci.FCI(mol, mf.mo_coeff)
#cisolver.nroots = 10000
#print('E(FCI) = %s' % cisolver.kernel()[0])

