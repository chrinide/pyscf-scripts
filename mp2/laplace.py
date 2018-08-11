#!/usr/bin/python

import os, numpy, ctypes
from pyscf import gto, scf, lib, ao2mo, dft
einsum = lib.einsum

os.environ['LAPLACE_ROOT'] = '/home/jluis/src/git/lmp2'
_loaderpath = '/home/jluis/src/git/lmp2/lib'
liblaplace = numpy.ctypeslib.load_library('liblaplace_minimax.so', _loaderpath)

mol = gto.Mole()
mol.basis = 'def2-tzvpd'
mol.atom = '''
O      0.000000      0.000000      0.118351
H      0.000000      0.761187     -0.469725
H      0.000000     -0.761187     -0.469725
'''
mol.verbose = 4
mol.spin = 0
mol.symmetry = 0
mol.charge = 0
mol.build()

# PBE0-2
mf = dft.RKS(mol)
mf.xc = '0.793701*hf + 0.206299*pbe, 0.5*pbe'
ehf = mf.kernel()

c_mp2 = 0.5
nao, nmo = mf.mo_coeff.shape
ncore = 0
nocc = mol.nelectron//2 - ncore
nvir = nmo - nocc - ncore
mo_core = mf.mo_coeff[:,:ncore]
mo_occ = mf.mo_coeff[:,ncore:ncore+nocc]
mo_vir = mf.mo_coeff[:,ncore+nocc:]

co = mo_occ
cv = mo_vir
eo = mf.mo_energy[ncore:ncore+nocc]
ev = mf.mo_energy[ncore+nocc:]
e_denom = 1.0 / (eo.reshape(-1, 1, 1, 1) - ev.reshape(-1, 1, 1) + eo.reshape(-1, 1) - ev)

eri_mo = ao2mo.general(mf._eri, (co,cv,co,cv), compact=False)
eri_mo = eri_mo.reshape(nocc,nvir,nocc,nvir)

rdm2_mp2 = numpy.zeros((nocc,nvir,nocc,nvir))
rdm2_mp2 = 2.0*einsum('iajb,iajb->iajb', eri_mo, e_denom)
rdm2_mp2 -= einsum('ibja,iajb->iajb', eri_mo, e_denom)
e_mp2 = einsum('iajb,iajb->', eri_mo, rdm2_mp2)
lib.logger.info(mf,'E(MP2) is %.8f' % (c_mp2*e_mp2))
lib.logger.info(mf,'E(MP2+HF) is %.8f' % (c_mp2*e_mp2+ehf))

lib.logger.info(mf,'Follow laplace transformed MP2')
npoints = 5
errmax = numpy.zeros(1)
xpnts = numpy.zeros(npoints)
wghts = numpy.zeros(npoints)
ymin = 2.0*(mf.mo_energy[(mol.nelectron//2)]-mf.mo_energy[mol.nelectron//2-1])
ymax = 2.0*(mf.mo_energy[nao-1]-mf.mo_energy[0])
lib.logger.info(mf,"Energy range %.8f %.8f" % (ymin, ymax))

liblaplace.laplace_points(errmax.ctypes.data_as(ctypes.c_void_p),
                          xpnts.ctypes.data_as(ctypes.c_void_p),
                          wghts.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(npoints), ctypes.c_double(ymin), ctypes.c_double(ymax))

lib.logger.info(mf,'Error on laplace data: %.8f' % errmax)
lib.logger.info(mf,'Weigths : %s' % wghts)
lib.logger.info(mf,'Time coordinates: %s' % xpnts)
      
emp2 = 0.0
for i in range(npoints):
    co = mo_occ*numpy.exp(+xpnts[i]*eo/2.0)
    cv = mo_vir*numpy.exp(-xpnts[i]*ev/2.0)
    eri_mo = ao2mo.general(mf._eri, (co,cv,co,cv), compact=False)
    eri_mo = eri_mo.reshape(nocc,nvir,nocc,nvir)
    tmp = (2.0*eri_mo - eri_mo.transpose(0,3,2,1))*eri_mo
    emp2 -= wghts[i]*numpy.einsum('iajb->',tmp)

lib.logger.info(mf,'E(MP2) is %.8f' % (c_mp2*emp2))
lib.logger.info(mf,'Energy diff: %.8f' % (abs(e_mp2-emp2)*c_mp2))
