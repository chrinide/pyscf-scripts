#!/usr/bin/env python

import numpy
from pyscf import gto, scf

mol = gto.Mole()
mol.atom = '''
  C    0.0000  0.0000  0.0000
  N    0.0000  0.0000  1.1674
'''
mol.basis = 'dzp'
mol.verbose = 4
mol.spin = 1
mol.symmetry = 1
mol.charge = 0
mol.build()

lmult = 0.1
diis_obj = scf.EDIIS()
diis_obj.diis_space = 20

def get_sfock(h1e, s1e, vhf, dm, cycle=0, diis=diis_obj):
    fock0 = old_get_fock(h1e, s1e, vhf, dm, cycle, diis)
    fock = numpy.zeros_like(fock0)
    mata = numpy.einsum('ik,kj->ij', s1e, dm[1])
    mata = numpy.einsum('ik,kj->ij', mata, s1e)
    matb = numpy.einsum('ik,kj->ij', s1e, dm[0])
    matb = numpy.einsum('ik,kj->ij', matb, s1e)
    fock[0] = fock0[0] - 2.0*lmult*mata
    fock[1] = fock0[1] - 2.0*lmult*matb
    return fock

mf = scf.UHF(mol)#.newton()
mf.conv_tol = 1e-6
mf.max_cycle = 100
mf.diis = diis_obj
old_get_fock = mf.get_fock
mf.get_fock = get_sfock
mf.kernel()

nocc_a, nocc_b = mf.nelec
s = mf.get_ovlp()
pa, pb = mf.make_rdm1()

# S^2 = SZ(SZ+1)+nb-tr(spbspa) --> single det
sz = (nocc_a - nocc_b)/2.0 
tmp1 = numpy.einsum('ik,kj->ij',s,pa)
tmp2 = numpy.einsum('ik,kj->ij',s,pb)
tr = numpy.einsum('ij,ji->',tmp1,tmp2)
ss = sz*(sz+1.0)+nocc_b-tr
print ("S^2",ss)

