#!/usr/bin/env python

import numpy
from pyscf import gto, scf, lib, ao2mo, df
from pyscf.tools import molden
einsum = lib.einsum

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
mol.symmetry = 1
mol.verbose = 4
mol.build()

mf = scf.RHF(mol)
mf = scf.density_fit(mf)
mf.auxbasis = 'aug-cc-pvdz-jkfit'
ehf = mf.kernel()

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
            
with_df = df.DF(mf.mol)
with_df.auxbasis = df.make_auxbasis(mf.mol, mp2fit=True)
with_df.kernel()

#naux = mf._cderi.shape[0]
naux = with_df._cderi.shape[0]
dferi = numpy.empty((naux,nao,nao))
for i in range(naux):
    #dferi[i] = lib.unpack_tril(mf._cderi[i])
    dferi[i] = lib.unpack_tril(with_df._cderi[i])
eri_mo = einsum('rj,Qrs->Qjs', co, dferi)
eri_mo = einsum('sb,Qjs->Qjb', cv, eri_mo)

vv_denom = -ev.reshape(-1,1)-ev
t2 = numpy.zeros((nocc,nvir,nocc,nvir))
for i in range(nocc):
    eps_i = eo[i]
    i_Qv = eri_mo[:, i, :].copy()
    for j in range(nocc):
        eps_j = eo[j]
        j_Qv = eri_mo[:, j, :].copy()
        viajb = einsum('Qa,Qb->ab', i_Qv, j_Qv)
        vibja = einsum('Qb,Qa->ab', i_Qv, j_Qv)
        div = 1.0/(eps_i + eps_j + vv_denom)
        t2[i,:,j,:] += 2.0*einsum('ab,ab->ab', viajb, div) 
        t2[i,:,j,:] -= 1.0*einsum('ab,ab->ab', vibja, div) 

e_mp2 = einsum('iajb,Qia->Qjb', t2, eri_mo)
e_mp2 = numpy.einsum('Qjb,Qjb->', e_mp2, eri_mo, optimize=True)
lib.logger.info(mf,"!*** E(MP2): %12.8f" % e_mp2)
lib.logger.info(mf,"!**** E(HF+MP2): %12.8f" % (e_mp2+ehf))

