#!/usr/bin/env python

import os, time, numpy, ctypes
from pyscf import gto, scf, lib, ao2mo
einsum = lib.einsum

_loaderpath = os.path.dirname(__file__)
libmp2 = numpy.ctypeslib.load_library('libmp2.so', _loaderpath)

mol = gto.Mole()
mol.basis = 'sto-3g'
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

mf = scf.RHF(mol)
mf.kernel()

ncore = 1
nao, nmo = mf.mo_coeff.shape
nocc = mol.nelectron//2 - ncore
nvir = nmo - nocc - ncore
norb = nocc + nvir
mo_core = mf.mo_coeff[:,:ncore]
mo_occ = mf.mo_coeff[:,ncore:ncore+nocc]
mo_vir = mf.mo_coeff[:,ncore+nocc:]
co = mo_occ
cv = mo_vir
eo = mf.mo_energy[ncore:ncore+nocc]
ev = mf.mo_energy[ncore+nocc:]
lib.logger.info(mf,"* Core orbitals: %d" % ncore)
lib.logger.info(mf,"* Virtual orbitals: %d" % (len(ev)))

t = time.time()
eri_mo = ao2mo.general(mf._eri, (co,cv,co,cv), compact=False)
eri_mo = eri_mo.reshape(nocc,nvir,nocc,nvir)
eri_mo = numpy.asarray(eri_mo, order='C')
t2 = numpy.zeros((nocc,nvir,nocc,nvir))
t2 = numpy.asarray(t2, order='C')
eorb = numpy.hstack((eo,ev))
eorb = numpy.asarray(eorb, order='C')
libmp2.mp2.restype = ctypes.c_double

e_mp2 = libmp2.mp2(t2.ctypes.data_as(ctypes.c_void_p), 
           eri_mo.ctypes.data_as(ctypes.c_void_p), 
           ctypes.c_int(norb), 
           ctypes.c_int(nocc), 
           eorb.ctypes.data_as(ctypes.c_void_p))

lib.logger.info(mf, 'E_MP2 : %.12f' % e_mp2)
lib.logger.info(mf, 'Total time taken : %.3f seconds' % (time.time()-t))

