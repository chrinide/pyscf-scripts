#!/usr/bin/env python

import numpy
from functools import reduce
from pyscf import gto, scf, lib, ao2mo
einsum = lib.einsum

mol = gto.Mole()
mol.basis = 'aug-cc-pvtz'
mol.atom = '''
O
H 1 1.1
H 1 1.1 2 104
'''
mol.charge = 0
mol.spin = 0
mol.symmetry = 1
mol.verbose = 4
mol.build()

mf = scf.RHF(mol)
ehf = mf.kernel()

nao, nmo = mf.mo_coeff.shape
ncore = 1
nocc = mol.nelectron//2 - ncore
nvir = nmo - nocc - ncore
lib.logger.info(mf,"* Core orbitals: %d" % ncore)
lib.logger.info(mf,"* Ocuppied orbitals: %d" % nocc)
lib.logger.info(mf,"* Virtual orbitals: %d" % nvir)

mo_core = mf.mo_coeff[:,:ncore]
mo_occ = mf.mo_coeff[:,ncore:ncore+nocc]
mo_vir = mf.mo_coeff[:,ncore+nocc:]
co = mo_occ
cv = mo_vir
c = numpy.hstack([co,cv])
ec = mf.mo_energy[:ncore]
eo = mf.mo_energy[ncore:ncore+nocc]
ev = mf.mo_energy[ncore+nocc:]
eps = numpy.hstack([eo,ev]) 
eri_mo = ao2mo.general(mf._eri, (c,c,c,c), compact=False)

ntot = nocc + nvir
eri_mo = eri_mo.reshape(ntot,ntot,ntot,ntot)
eri_mo = eri_mo.swapaxes(1, 2)

# Create occupied and virtual slices
o = slice(0, nocc)
v = slice(nocc, eri_mo.shape[0])

ep2_arr = []
for orbital in range(ntot):
    E = eps[orbital]
    ep2_conv = False
    for ep_iter in range(25):
        Eold = E

        # Build energy denominators
        epsilon1 = 1.0/(E + eo.reshape(-1, 1, 1) - ev.reshape(-1, 1) - ev)
        epsilon2 = 1.0/(E + ev.reshape(-1, 1, 1) - eo.reshape(-1, 1) - eo)

        # Compute sigma's
        tmp1 = (2.0*eri_mo[orbital, o, v, v] - eri_mo[o, orbital, v, v])
        sigma1 = numpy.einsum('rsa,ars,ars->', eri_mo[v, v, orbital, o], tmp1, epsilon1)
        tmp2 = (2.0*eri_mo[orbital, v, o, o] - eri_mo[v, orbital, o, o])
        sigma2 = numpy.einsum('abr,rab,rab->', eri_mo[o, o, orbital, v], tmp2, epsilon2)
        Enew = eps[orbital] + sigma1 + sigma2

        # Break if below threshold
        if abs(Enew - Eold) < 1.e-4:
            ep2_conv = True
            ep2_arr.append(Enew * 27.21138505)
            break

        # Build derivatives
        sigma_deriv1 = numpy.einsum('rsa,ars,ars->', eri_mo[v, v, orbital, o], tmp1, numpy.power(epsilon1, 2))
        sigma_deriv2 = numpy.einsum('abr,rab,rab->', eri_mo[o, o, orbital, v], tmp2, numpy.power(epsilon2, 2))
        deriv = -1.0*(sigma_deriv1 + sigma_deriv2)

        # Newton-Raphson update
        E = Eold - (Eold - Enew) / (1.0 - deriv)

    if ep2_conv is False:
        ep2_arr.append(E * 27.21138505)
        print('WARNING: EP2 for orbital HOMO - %d did not converged' % (orbital+ncore))

print("KP - Koopmans' Theorem")
print("EP2 - Electron Propagator 2\n")
print("Orbital         KP (eV)              EP2 (eV)")
print("----------------------------------------------")

KP_arr = eps*27.21138505

for orbital in range(ntot):
    kp_orb = ncore + orbital
    print("% 4d     % 16.4f    % 16.4f" % (kp_orb, KP_arr[orbital], ep2_arr[orbital]))

ep2_arr = numpy.asarray(ep2_arr)
ep2_arr = ep2_arr / 27.21138505
ep2_arr = numpy.hstack([ec,ep2_arr])
eo = ep2_arr[ncore:ncore+nocc]
ev = ep2_arr[ncore+nocc:]
eri_mo = ao2mo.general(mf._eri, (co,cv,co,cv), compact=False)
eri_mo = eri_mo.reshape(nocc,nvir,nocc,nvir)
e_denom = 1.0/(eo.reshape(-1,1,1,1)-ev.reshape(-1,1,1)+eo.reshape(-1,1)-ev)
t2 = numpy.zeros((nocc,nvir,nocc,nvir))
t2 = 2.0*einsum('iajb,iajb->iajb', eri_mo, e_denom)
t2 -= einsum('ibja,iajb->iajb', eri_mo, e_denom)
e_mp2 = numpy.einsum('iajb,iajb->', eri_mo, t2, optimize=True)
lib.logger.info(mf,"!*** E(MP2): %12.8f" % e_mp2)
lib.logger.info(mf,"!**** E(HF+MP2): %12.8f" % (e_mp2+ehf))

