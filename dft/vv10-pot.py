#!/usr/bin/env python

#https://github.com/psi4/psi4numpy/blob/master/Tutorials/04_Density_Functional_Theory/4d_VV10.ipynb

import numpy
from pyscf import gto, scf, lib, dft

mol = gto.Mole()
mol.verbose = 4
mol.atom = '''
He 0 0 -5
He 0 0  5
'''
mol.basis = 'aug-cc-pvdz'
mol.symmetry = 0
mol.build()

mf = dft.RKS(mol)
mf.xc = 'rPW86,PBE'
mf.grids.becke_scheme = dft.stratmann
mf.grids.atom_grid = {(20, 110)}
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
lib.logger.info(mf, 'Rho = %.12f' % numpy.einsum('i,i->', rho[0], weights))
ex, vx = dft.libxc.eval_xc('rPW86,', rho)[:2]
ec, vc = dft.libxc.eval_xc(',PBE', rho)[:2]
lib.logger.info(mf, 'Exc = %.12f' % numpy.einsum('i,i,i->', ex+ec, rho[0], weights))

coef_C = 0.0093
coef_B = 5.9
coef_beta = 1.0/32.0 * (3.0/(coef_B**2.0))**(3.0/4.0)
kappa_pref = coef_B * (1.5*numpy.pi)/((9.0*numpy.pi)**(1.0/6.0))
const = 4.0/3.0 * numpy.pi
vv10_e = 0.0

nao = mf.mo_occ.shape[0]
vpot = numpy.zeros((nao,nao))

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
    #
    F_U = kernel12*((1.0/g) + (1.0/(g+gp)))
    F_W = F_U*R

    # Energy 
    kernel = kernel12.sum()
    vv10_e += weigth1*rho1*(coef_beta + 0.5*kernel)

    # Potential at each point
    phi_U = F_U.sum()
    phi_W = F_W.sum()
    kappa_dn = kappa1/(6.0*rho1)
    w0_dgamma = coef_C * gamma1/(W01*rho1**4.0)
    w0_drho = 2.0/W01 * (numpy.pi/3.0 - coef_C * (gamma1*gamma1)/(rho1**5.0))

    v_rho = coef_beta + kernel + rho1*(kappa_dn*phi_U + w0_drho*phi_W)
    v_rho *= 0.5

    v_gamma = rho1 * w0_dgamma * phi_W
    tmp_grid = 2.0*weigth1*v_gamma

    tmp = numpy.einsum('i,j->ij', ao[0,idx1,:], v_rho*weigth1*ao[0,idx1,:])
    tmp += numpy.einsum('i,j->ij', ao[1,idx1,:], tmp_grid*rho[1,idx1]*ao[0,idx1,:])
    tmp += numpy.einsum('i,j->ij', ao[2,idx1,:], tmp_grid*rho[2,idx1]*ao[0,idx1,:])
    tmp += numpy.einsum('i,j->ij', ao[3,idx1,:], tmp_grid*rho[3,idx1]*ao[0,idx1,:])

    vpot += tmp + tmp.T

print('VV10 = %.12f' % vv10_e)
for i in range(nao):
    print vpot[i,:]

