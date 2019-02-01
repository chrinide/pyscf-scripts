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
dm = mf.get_init_guess() + 0.2j
mf.kernel(dm)

nao, nmo = mf.mo_coeff.shape
eri_mo = ao2mo.kernel(mol, mf.mo_coeff[:,:nmo], compact=False, intor='int2e_spinor')
eri_mo = eri_mo.reshape(nmo,nmo,nmo,nmo)
h1 = reduce(numpy.dot, (mf.mo_coeff[:,:nmo].conj().T, mf.get_hcore(), mf.mo_coeff[:,:nmo]))

h = numpy.zeros((2,2), dtype=numpy.complex128)
h[0,0] = 2.0*h1[0,0]+eri_mo[0,0,0,0]
h[1,0] = eri_mo[0,2,0,2]
h[1,1] = 2.0*h1[2,2]+eri_mo[2,2,2,2]
h[0,1] = eri_mo[2,0,2,0]
e,c = numpy.linalg.eigh(h)
e += mf.energy_nuc()
print('E(GVB-PP) = %s' % e)

