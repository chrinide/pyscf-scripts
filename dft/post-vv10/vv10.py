#!/usr/bin/env python

import os, time, numpy, ctypes
from pyscf import gto, scf, lib, dft

_loaderpath = os.path.dirname(__file__)
libvv10 = numpy.ctypeslib.load_library('libvv10.so', _loaderpath)

mol = gto.Mole()
mol.verbose = 4
mol.atom = '''
O
H 1 1.1
H 1 1.1 2 104.0
'''
mol.basis = 'aug-cc-pvdz'
mol.symmetry = 1
mol.build()

mf = dft.RKS(mol)
mf.xc = 'rpw86,pbe'
mf.grids.level = 1
mf.kernel()

dm = mf.make_rdm1()
coords = mf.grids.coords
weights = mf.grids.weights
ngrids = len(weights)
ao = dft.numint.eval_ao(mol, coords, deriv=1)
rho = dft.numint.eval_rho(mol, ao, dm, xctype='GGA')
gnorm2 = numpy.zeros(ngrids)
for i in range(ngrids):
    gnorm2[i] = numpy.linalg.norm(rho[-3:,i])**2
lib.logger.info(mf,'Rho = %.12f' % numpy.einsum('i,i->', rho[0], weights))
ex, vx = dft.libxc.eval_xc('rPW86,', rho)[:2]
ec, vc = dft.libxc.eval_xc(',PBE', rho)[:2]
lib.logger.info(mf, 'Exc = %.12f' % numpy.einsum('i,i,i->', ex+ec, rho[0], weights))

coef_C = 0.0093
coef_B = 5.9
coef_beta = 1.0/32.0 * (3.0/(coef_B**2.0))**(3.0/4.0)
kappa_pref = coef_B * (1.5*numpy.pi)/((9.0*numpy.pi)**(1.0/6.0))
const = 4.0/3.0 * numpy.pi
vv10_e = 0.0

t = time.time()
for idx1 in range(ngrids):
    point1 = coords[idx1,:]
    rho1 = rho[0,idx1]
    weigth1 = weights[idx1]
    gamma1 = gnorm2[idx1]
    Wp1 = const*rho1
    Wg1 = coef_C * ((gamma1/(rho1*rho1))**2.0)
    W01 = numpy.sqrt(Wg1 + Wp1)
    kappa1 = rho1**(1.0/6.0)*kappa_pref
    #
    R =  (point1[0]-coords[:,0])**2
    R += (point1[1]-coords[:,1])**2
    R += (point1[2]-coords[:,2])**2
    Wp2 = const*rho[0]
    Wg2 = coef_C * ((gnorm2/(rho[0]*rho[0]))**2.0)
    W02 = numpy.sqrt(Wg2 + Wp2)
    kappa2 = rho[0]**(1.0/6.0)*kappa_pref
    g = W01*R + kappa1
    gp = W02*R + kappa2
    kernel12 = -1.5*weights*rho[0]/(g*gp*(g+gp))

    # Energy 
    kernel = kernel12.sum()
    vv10_e += weigth1*rho1*(coef_beta + 0.5*kernel)
lib.logger.info(mf, 'VV10 = %.12f' % vv10_e)
lib.logger.info(mf, 'Total time taken VV10: %.3f seconds' % (time.time()-t))
   
t = time.time()
libvv10.vv10.restype = ctypes.c_double
ev = libvv10.vv10(ctypes.c_int(ngrids),
             ctypes.c_double(coef_C),
             ctypes.c_double(coef_B),
             coords.ctypes.data_as(ctypes.c_void_p),
             rho.ctypes.data_as(ctypes.c_void_p),
             weights.ctypes.data_as(ctypes.c_void_p),
             gnorm2.ctypes.data_as(ctypes.c_void_p))
lib.logger.info(mf, 'VV10 = %.12f' % ev)
lib.logger.info(mf, 'Total time taken VV10 (C): %.3f seconds' % (time.time()-t))
