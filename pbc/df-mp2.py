#!/usr/bin/env python

import numpy, utils
from pyscf import lib
from pyscf.pbc import gto, scf, cc, df, cc, mp

name = 'df-mp2'

cell = gto.Cell()
cell.unit = 'A'
cell.a = [[8.,0.,0.],
          [0.,1.,0.],
          [0.,0.,1.]] 
cell.atom= '''
 H 0 0 0
 H 1 0 0
 H 2 0 0
 H 3 0 0
 H 4 0 0
 H 5 0 0
 H 6 0 0
 H 7 0 0
'''
cell.basis = 'cc-pvtz'
cell.dimension = 1
cell.verbose = 4
cell.build()
 
gdf = df.GDF(cell)

mf = scf.RHF(cell)
mf.with_df = gdf
mf.with_df.auxbasis = 'cc-pvtz-jkfit'
#mf.with_df.mesh = [10,10,14] 
#mf.exxdiv = None
mf.max_cycle = 150
mf.chkfile = name+'.chk'
#mf.with_df._cderi_to_save = name+'_eri.h5'
#mf.with_df._cderi = name+'_eri.h5' 
#mf = mole_scf.addons.remove_linear_dep_(mf)
#mf.__dict__.update(scf.chkfile.load(name+'.chk', 'scf'))
#dm = mf.make_rdm1()
ehf = mf.kernel()

ncore = 0 
nao, nmo = mf.mo_coeff.shape
nocc = cell.nelectron//2 - ncore
nvir = nmo - nocc - ncore
mo_core = mf.mo_coeff[:,:ncore]
mo_occ = mf.mo_coeff[:,ncore:ncore+nocc]
mo_vir = mf.mo_coeff[:,ncore+nocc:]
co = mo_occ
cv = mo_vir
eo = mf.mo_energy[ncore:ncore+nocc]
ev = mf.mo_energy[ncore+nocc:]
lib.logger.info(mf,"\n* GAMMA point MP2 ")
lib.logger.info(mf,"* Core orbitals: %d" % ncore)
lib.logger.info(mf,"* Virtual orbitals: %d" % (len(ev)))

naux = gdf.get_naoaux()
nao = cell.nao
dferi = numpy.zeros((naux,nao,nao))
kpt = numpy.zeros(3)
p1 = 0
for LpqR, LpqI, sign in gdf.sr_loop((kpt,kpt), compact=False):
    p0, p1 = p1, p1 + LpqR.shape[0]
    dferi[p0:p1] = LpqR.reshape(-1,nao,nao)

eri_mo = lib.einsum('rj,Qrs->Qjs', co, dferi)
eri_mo = lib.einsum('sb,Qjs->Qjb', cv, eri_mo)
vv_denom = -ev.reshape(-1,1)-ev
t2 = numpy.zeros((nocc,nvir,nocc,nvir))
for i in range(nocc):
    eps_i = eo[i]
    i_Qv = eri_mo[:, i, :].copy()
    for j in range(nocc):
        eps_j = eo[j]
        j_Qv = eri_mo[:, j, :].copy()
        viajb = lib.einsum('Qa,Qb->ab', i_Qv, j_Qv)
        vibja = lib.einsum('Qb,Qa->ab', i_Qv, j_Qv)
        div = 1.0/(eps_i + eps_j + vv_denom)
        t2[i,:,j,:] += 2.0*lib.einsum('ab,ab->ab', viajb, div) 
        t2[i,:,j,:] -= 1.0*lib.einsum('ab,ab->ab', vibja, div) 

e_mp2 = lib.einsum('iajb,Qia->Qjb', t2, eri_mo)
e_mp2 = numpy.einsum('Qjb,Qjb->', e_mp2, eri_mo, optimize=True)
lib.logger.info(mf,"!*** E(MP2): %12.8f" % e_mp2)
lib.logger.info(mf,"!**** E(HF+MP2): %12.8f" % (e_mp2+ehf))

cv, ev = utils.getdffno(mf,ncore,eri_mo,thresh_vir=1e-4)
lib.logger.info(mf,"* FNO GAMMA point MP2 ")
nvir = len(ev)
t2 = numpy.zeros((nocc,nvir,nocc,nvir))
eri_mo = lib.einsum('rj,Qrs->Qjs', co, dferi)
eri_mo = lib.einsum('sb,Qjs->Qjb', cv, eri_mo)
vv_denom = -ev.reshape(-1,1)-ev
for i in range(nocc):
    eps_i = eo[i]
    i_Qv = eri_mo[:, i, :].copy()
    for j in range(nocc):
        eps_j = eo[j]
        j_Qv = eri_mo[:, j, :].copy()
        viajb = lib.einsum('Qa,Qb->ab', i_Qv, j_Qv)
        vibja = lib.einsum('Qb,Qa->ab', i_Qv, j_Qv)
        div = 1.0 / (eps_i + eps_j + vv_denom)
        t2[i,:,j,:] += 2.0*lib.einsum('ab,ab->ab', viajb, div) 
        t2[i,:,j,:] -= 1.0*lib.einsum('ab,ab->ab', vibja, div) 
e_mp2 = lib.einsum('iajb,Qia->Qjb', t2, eri_mo)
e_mp2 = numpy.einsum('Qjb,Qjb->', e_mp2, eri_mo, optimize=True)
lib.logger.info(mf,"!*** FNO E(MP2): %12.8f" % e_mp2)
lib.logger.info(mf,"!**** FNO E(HF+MP2): %12.8f" % (e_mp2+ehf))

#pt = mp.RMP2(mf)
#pt.frozen = ncore
#pt.kernel()
