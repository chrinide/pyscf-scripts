#!/usr/bin/env python

import numpy, sys
sys.path.append('../tools')
from pyscf import gto, scf, lib, ao2mo
from pyscf.tools import molden
import utils
einsum = lib.einsum

name = 'ch4'

mol = gto.Mole()
mol.basis = 'aug-cc-pvdz'
mol.atom = '''
C  0.0000  0.0000  0.0000
H  0.6276  0.6276  0.6276
H  0.6276 -0.6276 -0.6276
H -0.6276  0.6276 -0.6276
H -0.6276 -0.6276  0.6276
'''
mol.charge = 0
mol.spin = 0
mol.symmetry = 'c1'
mol.verbose = 4
mol.build()

mf = scf.RHF(mol)
mf.chkfile = name+'.chk'
mf.level_shift = 0.5
mf = scf.density_fit(mf)
mf.auxbasis = 'aug-cc-pvdz-jkfit'
mf = scf.addons.remove_linear_dep_(mf)
mf = scf.newton(mf)
mf.kernel()
dm = mf.make_rdm1()
mf.level_shift = 0.0
ehf = mf.kernel(dm)
 
ncore = 1
nao, nmo = mf.mo_coeff.shape
nocc = mol.nelectron//2 - ncore
nvir = nmo - nocc - ncore
mo_core = mf.mo_coeff[:,:ncore]
mo_occ = mf.mo_coeff[:,ncore:ncore+nocc]
mo_vir = mf.mo_coeff[:,ncore+nocc:]
co = mo_occ
cv = mo_vir
eo = mf.mo_energy[ncore:ncore+nocc]
ev = mf.mo_energy[ncore+nocc:]
lib.logger.info(mf,"* Core orbitals: %d" % ncore)
lib.logger.info(mf,"* Virtual orbitals: %d" % (len(ev)))

naux = mf._cderi.shape[0]
dferi = numpy.empty((naux,nao,nao))
for i in range(naux):
    dferi[i] = lib.unpack_tril(mf._cderi[i])
eri_mo = einsum('rj,Qrs->Qjs', co, dferi)
eri_mo = einsum('sb,Qjs->Qjb', cv, eri_mo)
cv,ev = utils.getdffno(mf,ncore,eri_mo,thresh_vir=1e-4) 

nvir = len(ev)
t2 = numpy.zeros((nocc,nvir,nocc,nvir))
eri_mo = einsum('rj,Qrs->Qjs', co, dferi)
eri_mo = einsum('sb,Qjs->Qjb', cv, eri_mo)
vv_denom = -ev.reshape(-1,1)-ev
for i in range(nocc):
    eps_i = eo[i]
    i_Qv = eri_mo[:, i, :].copy()
    for j in range(nocc):
        eps_j = eo[j]
        j_Qv = eri_mo[:, j, :].copy()
        viajb = einsum('Qa,Qb->ab', i_Qv, j_Qv)
        vibja = einsum('Qb,Qa->ab', i_Qv, j_Qv)
        div = 1.0 / (eps_i + eps_j + vv_denom)
        t2[i,:,j,:] += 2.0*einsum('ab,ab->ab', viajb, div) 
        t2[i,:,j,:] -= 1.0*einsum('ab,ab->ab', vibja, div) 
e_mp2 = einsum('iajb,Qia->Qjb', t2, eri_mo)
e_mp2 = numpy.einsum('Qjb,Qjb->', e_mp2, eri_mo, optimize=True)
lib.logger.info(mf,"!*** E(MP2): %12.8f" % e_mp2)
lib.logger.info(mf,"!**** E(HF+MP2): %12.8f" % (e_mp2+ehf))

