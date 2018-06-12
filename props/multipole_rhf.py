#!/usr/bin/env python

# https://aip.scitation.org/doi/abs/10.1063/1.460293

import numpy
from pyscf import gto, scf, lib

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
dm = mf.make_rdm1()

unit = 2.541746
origin = ([0.0,0.0,0.0])
charges = mol.atom_charges()
coords  = mol.atom_coords()
mol.set_common_orig(origin)

r2 = mol.intor_symmetric('int1e_r2')
r2 = numpy.einsum('ij,ji->', r2, dm)
lib.logger.info(mf,'Electronic spatial extent <R**2> (au): %.4f', r2)

lib.logger.info(mf,'* Multipoles in the independent field-basis, Gauge -> (0,0,0)')
lib.logger.info(mf,'* The electronic part is considered as negative, while positive for the nuclear part')
lib.logger.info(mf,'* This is the reverse criteria used in Gaussian')

ao_dip = mol.intor_symmetric('int1e_r', comp=3)
el_dip = numpy.einsum('xij,ji->x', ao_dip, dm)
lib.logger.info(mf,'Electronic Dipole moment(X, Y, Z, Debye): %.4f, %.4f, %.4f', *el_dip*unit)
nucl_dip = numpy.einsum('i,ix->x', charges, coords)
lib.logger.info(mf,'Nuclear Dipole moment(X, Y, Z, Debye): %.4f, %.4f, %.4f', *nucl_dip*unit)
mol_dip = (nucl_dip - el_dip) * unit
lib.logger.info(mf,'Total Dipole moment(X, Y, Z, Debye): %.4f, %.4f, %.4f', *mol_dip)

lib.logger.info(mf,'Quadrupole moments (Debye-Angs)')
rr = mol.intor_symmetric('int1e_rr', comp=9).reshape(3,3,nao,nao)
rr = -1.0*numpy.einsum('xyij,ji->xy', rr, dm)
rr += numpy.einsum('z,zx,zy->xy', charges, coords, coords)
rr = rr*unit*lib.param.BOHR
lib.logger.info(mf,'Total Quadrupole moments (XX, YY, ZZ): %.4f, %.4f, %.4f', \
rr[0,0], rr[1,1], rr[2,2])
lib.logger.info(mf,'Total Quadrupole moments (XY, XZ, YZ): %.4f, %.4f, %.4f', \
rr[0,1], rr[0,2], rr[1,2])
 
lib.logger.info(mf,'Octupole moments (Debye-Angs**2)')
rrr = mol.intor_symmetric('int1e_rrr', comp=27).reshape(3,3,3,nao,nao)
rrr = -1.0*numpy.einsum('xyzij,ji->xyz', rrr, dm)
rrr += numpy.einsum('z,zx,zy,zk->xyk', charges, coords, coords, coords)
rrr = rrr*unit*lib.param.BOHR**2
lib.logger.info(mf,'Total Octupole moments (XXX, YYY, ZZZ, XYY): %.4f, %.4f, %.4f, %.4f', \
rrr[0,0,0], rrr[1,1,1], rrr[2,2,2], rrr[0,1,1])
lib.logger.info(mf,'Total Octupole moments (XXY, XXZ, XZZ, YZZ): %.4f, %.4f, %.4f, %.4f', \
rrr[0,0,1], rrr[0,0,2], rrr[0,2,2], rrr[1,2,2])
lib.logger.info(mf,'Total Octupole moments (YYZ, XYZ): %.4f, %.4f', rrr[1,1,2], rrr[0,1,2])

lib.logger.info(mf,'Hexadecapole moments (Debye-Angs**3)')
rrrr = mol.intor_symmetric('int1e_rrrr', comp=81).reshape(3,3,3,3,nao,nao)
rrrr = -1.0*numpy.einsum('xyzwij,ji->xyzw', rrrr, dm)
rrrr += numpy.einsum('z,zx,zy,zk,zw->xykw', charges, coords, coords, coords, coords)
rrrr = rrrr*unit*lib.param.BOHR**3
lib.logger.info(mf,'Total Hexadecapole moments (XXXX, YYYY, ZZZZ, XXXY): %.4f, %.4f, %.4f, %.4f', \
rrrr[0,0,0,0], rrrr[1,1,1,1], rrrr[2,2,2,2], rrrr[0,0,0,1])
lib.logger.info(mf,'Total Hexadecapole moments (XXXZ, YYYX, YYYZ, ZZZX): %.4f, %.4f, %.4f, %.4f', \
rrrr[0,0,0,2], rrrr[1,1,1,0], rrrr[1,1,1,2], rrrr[2,2,2,0])
lib.logger.info(mf,'Total Hexadecapole moments (ZZZY, XXYY, XXZZ, YYZZ): %.4f, %.4f, %.4f, %.4f', \
rrrr[2,2,2,1], rrrr[0,0,1,1], rrrr[0,0,2,2], rrrr[1,1,2,2])
lib.logger.info(mf,'Total Hexadecapole moments (XXYZ, YYXZ, ZZXY): %.4f, %.4f, %.4f', \
rrrr[0,0,1,2], rrrr[1,1,0,2], rrrr[2,2,0,1])

# Gaussian 09 reference data
# Electronic spatial extent (au):  <R**2>=             39.7795
# Dipole moment (field-independent basis, Debye):
#    X=              0.0000    Y=              0.0000    Z=             -0.2440  Tot=              0.2440
# Quadrupole moment (field-independent basis, Debye-Ang):
#   XX=            -10.2080   YY=            -10.2080   ZZ=            -12.2995
#   XY=              0.0000   XZ=              0.0000   YZ=              0.0000
# Traceless Quadrupole moment (field-independent basis, Debye-Ang):
#   XX=              0.6971   YY=              0.6971   ZZ=             -1.3943
#   XY=              0.0000   XZ=              0.0000   YZ=              0.0000
# Octapole moment (field-independent basis, Debye-Ang**2):
#  XXX=              0.0000  YYY=              0.0000  ZZZ=              6.2801  XYY=              0.0000
#  XXY=              0.0000  XXZ=              1.0409  XZZ=              0.0000  YZZ=              0.0000
#  YYZ=              1.0409  XYZ=              0.0000
# Hexadecapole moment (field-independent basis, Debye-Ang**3):
# XXXX=             -9.1209 YYYY=             -9.1209 ZZZZ=            -37.0722 XXXY=              0.0000
# XXXZ=              0.0000 YYYX=              0.0000 YYYZ=              0.0000 ZZZX=              0.0000
# ZZZY=              0.0000 XXYY=             -3.0403 XXZZ=             -7.0154 YYZZ=             -7.0154
# XXYZ=              0.0000 YYXZ=              0.0000 ZZXY=              0.0000
