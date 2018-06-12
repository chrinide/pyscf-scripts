#!/usr/bin/env python

# https://aip.scitation.org/doi/abs/10.1063/1.460293

import numpy, sys
from pyscf import gto, scf, lib, cc

mol = gto.Mole()
mol.basis = 'aug-cc-pvtz'
mol.atom = '''
C      0.000000      0.000000     -0.642367
O      0.000000      0.000000      0.481196
 '''
mol.verbose = 4
mol.spin = 0
mol.symmetry = 1
mol.charge = 0
mol.build()

nao = mol.nao_nr()
mf = scf.RHF(mol)
energy = mf.kernel()

ncore = 2
mcc = cc.CCSD(mf)
mcc.direct = 1
mcc.diis_space = 10
mcc.frozen = ncore
mcc.conv_tol = 1e-6
mcc.conv_tol_normt = 1e-6
mcc.max_cycle = 150
mcc.kernel()

t1norm = numpy.linalg.norm(mcc.t1)
t1norm = t1norm/numpy.sqrt(mol.nelectron-ncore*2)
lib.logger.info(mcc,"* T1 norm should be les than 0.02")
lib.logger.info(mcc,"* T1 norm : %12.6f" % t1norm)

nao, nmo = mf.mo_coeff.shape
rdm1 = mcc.make_rdm1()

unit = 2.541746
origin = ([0.0,0.0,0.0])
charges = mol.atom_charges()
coords  = mol.atom_coords()
mol.set_common_orig(origin)

# MO basis
r2 = mol.intor_symmetric('int1e_r2')
r2 = reduce(numpy.dot, (mf.mo_coeff.T, r2, mf.mo_coeff))
r2 = numpy.einsum('ij,ji->', r2, rdm1)
lib.logger.info(mf,'Electronic spatial extent <R**2> (au): %.4f', r2)

# AO basis
lib.logger.info(mf,'* Multipoles in the independent field-basis, Gauge -> (0,0,0)')
lib.logger.info(mf,'* The electronic part is considered as negative, while positive for the nuclear part')
lib.logger.info(mf,'* This is the reverse criteria used in Gaussian')
dm = reduce(lib.dot, (mf.mo_coeff,rdm1,mf.mo_coeff.T))
ao_dip = mol.intor_symmetric('int1e_r', comp=3)
el_dip = numpy.einsum('xij,ji->x', ao_dip, dm)
lib.logger.info(mf,'Electronic Dipole moment(X, Y, Z, Debye): %.4f, %.4f, %.4f', *el_dip*unit)
nucl_dip = numpy.einsum('i,ix->x', charges, coords)
lib.logger.info(mf,'Nuclear Dipole moment(X, Y, Z, Debye): %.4f, %.4f, %.4f', *nucl_dip*unit)
mol_dip = (nucl_dip - el_dip) * unit
lib.logger.info(mf,'Total Dipole moment(X, Y, Z, Debye): %.4f, %.4f, %.4f', *mol_dip)

