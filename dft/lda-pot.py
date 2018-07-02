#!/usr/bin/env python

import numpy
from pyscf import gto, scf, lib, dft

mol = gto.Mole()
mol.verbose = 4
mol.atom = '''
He 0 0 -5
He 0 0  5
'''
mol.basis = 'aug-cc-pvdz'
mol.symmetry = 1
mol.build()

mf = dft.RKS(mol)
mf.grids.radi_method = dft.mura_knowles
mf.grids.becke_scheme = dft.stratmann
mf.grids.level = 1
mf.kernel()

dm = mf.make_rdm1()
coords = mf.grids.coords
weights = mf.grids.weights
ngrids = len(weights)
nao = len(mf.mo_occ)
ao = dft.numint.eval_ao(mol, coords, deriv=0)
rho = dft.numint.eval_rho(mol, ao, dm, xctype='LDA')
exc, vxc = dft.libxc.eval_xc('LDA,VWN', rho)[:2]
lib.logger.info(mf, 'Rho = %.12f' % numpy.einsum('i,i->', rho, weights))
lib.logger.info(mf, 'Exc = %.12f' % numpy.einsum('i,i,i->', exc, rho, weights))
vxc = vxc[0]
aointer = numpy.einsum('pi,p->pi', ao, 0.5*weights*vxc)
vmat = numpy.einsum('pi,pj->ij', aointer, ao)
vmat = vmat + vmat.T

