#!/usr/bin/env python

import numpy
from functools import reduce
from pyscf import scf, gto, mp, lib, ao2mo, mp
from pyscf.tools import molden
einsum = lib.einsum

mol = gto.Mole()
mol.basis = '6-31g'
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

mf = scf.UHF(mol)
ehf = mf.kernel()
nao,nmo = mf.mo_coeff[0].shape

nelec = mol.nelectron
nelecb = nelec//2
neleca = nelec - nelecb
nocca = neleca
nvira = nmo - nocca
noccb = nelecb
nvirb = nmo - noccb

ca = mf.mo_coeff[0]
cb = mf.mo_coeff[1]
nmoa = ca.shape[1]
nmob = cb.shape[1]
eoa = mf.mo_energy[0][:nocca] 
eob = mf.mo_energy[1][:noccb]  
eva = mf.mo_energy[0][nocca:] 
evb = mf.mo_energy[1][noccb:]  

# Transform
t2aa = numpy.zeros((nvira,nocca,nvira,nocca))
eri_aa = ao2mo.kernel(mf._eri, ca, compact=False).reshape([nmoa]*4)
eri_aa = eri_aa - eri_aa.transpose(0,3,2,1)
e_aa = 1.0/(-eva.reshape(-1,1,1,1)+eoa.reshape(-1,1,1)-eva.reshape(-1,1)+eoa)

t2ab = numpy.zeros((nvira,nocca,nvirb,noccb))
eri_ab = ao2mo.kernel(mf._eri, (ca,ca,cb,cb), compact=False)
eri_ab = eri_ab.reshape([nmoa,nmoa,nmob,nmob])
e_ab = 1.0/(-eva.reshape(-1,1,1,1)+eoa.reshape(-1,1,1)-evb.reshape(-1,1)+eob)

t2bb = numpy.zeros((nvirb,noccb,nvirb,noccb))
eri_bb = ao2mo.kernel(mf._eri, cb, compact=False).reshape([nmob]*4)
eri_bb = eri_bb - eri_bb.transpose(0,3,2,1)
e_bb = 1.0/(-evb.reshape(-1,1,1,1)+eob.reshape(-1,1,1)-evb.reshape(-1,1)+eob)

oa = slice(0, nocca)
va = slice(nocca, None)
ob = slice(0, noccb)
vb = slice(noccb, None)

e1  = 0.25*numpy.einsum('iajb,aibj->', eri_aa[oa,va,oa,va]**2, e_aa)
e1 += 1.00*numpy.einsum('iaJB,aiBJ->', eri_ab[oa,va,ob,vb]**2, e_ab)
e1 += 0.25*numpy.einsum('iajb,aibj->', eri_bb[ob,vb,ob,vb]**2, e_bb)
lib.logger.info(mf,'MP2 energy %5.15f', e1)

e_cepa0 = 0.0
maxiter = 50
for it in range(maxiter+1):
    e_old = e_cepa0

    # Alpha-Alpha
    mp2_aa = eri_aa[va, oa, va, oa]

    cepa1_aa = 0.5*numpy.einsum('acbd,cidj->aibj', eri_aa[va,va,va,va], t2aa)
    cepa2_aa = 0.5*numpy.einsum('kilj,akbl->aibj', eri_aa[oa,oa,oa,oa], t2aa)

    cepa3a_aa = numpy.einsum('aikc,bjck->aibj', eri_aa[va,oa,oa,va], t2aa)
    cepa3b_aa = -cepa3a_aa.transpose(2,1,0,3)
    cepa3c_aa = -cepa3a_aa.transpose(0,3,2,1)
    cepa3d_aa = cepa3a_aa.transpose(2,3,0,1)
    cepa3_aa = cepa3a_aa + cepa3b_aa + cepa3c_aa + cepa3d_aa

    cepa3a_ab = numpy.einsum('aiKC,bjCK->aibj', eri_ab[va,oa,ob,vb], t2ab)
    cepa3b_ab = -cepa3a_ab.transpose(2,1,0,3)
    cepa3c_ab = -cepa3a_ab.transpose(0,3,2,1)
    cepa3d_ab = cepa3a_ab.transpose(2,3,0,1)
    cepa3_ab = cepa3a_ab + cepa3b_ab + cepa3c_ab + cepa3d_ab

    t2aa_new = e_aa*(cepa1_aa + cepa2_aa + cepa3_aa + cepa3_ab + mp2_aa)

    # Beta-Beta
    mp2_bb = eri_bb[vb, ob, vb, ob]

    cepa1_bb = 0.5*numpy.einsum('ACBD,CIDJ->AIBJ', eri_bb[vb,vb,vb,vb], t2bb)
    cepa2_bb = 0.5*numpy.einsum('KILJ,AKBL->AIBJ', eri_bb[ob,ob,ob,ob], t2bb)

    cepa3a_bb = numpy.einsum('AIKC,BJCK->AIBJ', eri_bb[vb,ob,ob,vb], t2bb)
    cepa3b_bb = -cepa3a_bb.transpose(2,1,0,3)
    cepa3c_bb = -cepa3a_bb.transpose(0,3,2,1)
    cepa3d_bb = cepa3a_bb.transpose(2,3,0,1)
    cepa3_bb = cepa3a_bb + cepa3b_bb + cepa3c_bb + cepa3d_bb

    cepa3a_ab = numpy.einsum('AIkc,BJck->AIBJ', eri_ab[va,oa,ob,vb], t2ab)
    cepa3b_ab = -cepa3a_ab.transpose(2,1,0,3)
    cepa3c_ab = -cepa3a_ab.transpose(0,3,2,1)
    cepa3d_ab = cepa3a_ab.transpose(2,3,0,1)
    cepa3_ab = cepa3a_ab + cepa3b_ab + cepa3c_ab + cepa3d_ab

    t2bb_new = e_bb*(cepa1_bb + cepa2_bb + cepa3_bb + cepa3_ab + mp2_bb)

    # Alpha-Beta == Beta-Alpha 
    mp2_ab = eri_ab[va, oa, vb, ob]
    cepa1_ab =  numpy.einsum("acBD,ciDJ->aiBJ", eri_ab[va,va,vb,vb], t2ab)
    cepa2_ab =  numpy.einsum("kiLJ,akBL->aiBJ", eri_ab[oa,oa,ob,ob], t2ab)
    cepa3_ab =  numpy.einsum("aikc,BJck->aiBJ", eri_aa[va,oa,oa,va], t2ab)
    cepa4_ab =  numpy.einsum("aiKC,BJCK->aiBJ", eri_ab[va,oa,ob,vb], t2bb)
    cepa5_ab = -numpy.einsum("acKJ,ciBK->aiBJ", eri_ab[va,va,ob,ob], t2ab)
    cepa6_ab = -numpy.einsum("kiBC,akCJ->aiBJ", eri_ab[oa,oa,vb,vb], t2ab)
    cepa7_ab =  numpy.einsum("BJkc,aick->aiBJ", eri_ab[vb,ob,oa,va], t2aa)
    cepa8_ab =  numpy.einsum("BJKC,aiCK->aiBJ", eri_bb[vb,ob,ob,vb], t2ab)
    
    t2ab_new = e_ab*(cepa1_ab + cepa2_ab + cepa3_ab + cepa4_ab +cepa5_ab + \
               cepa6_ab + cepa7_ab + cepa8_ab + mp2_ab)

    e_cepa0 = 0.25*numpy.einsum('iajb,aibj->', eri_aa[oa,va,oa,va], t2aa_new)
    e_cepa0 += 0.25*numpy.einsum('IAJB,AIBJ->', eri_bb[ob,vb,ob,vb], t2bb_new)
    e_cepa0 += numpy.einsum('iaJB,aiBJ->', eri_ab[oa,va,ob,vb], t2ab_new)

    t2aa = t2aa_new
    t2ab = t2ab_new
    t2bb = t2bb_new

    de = e_cepa0 - e_old
    lib.logger.info(mf,'Iteration %3d: energy = %4.12f de = %1.5e' % (it, e_cepa0, de))
    if abs(de) < 1.e-8:
        lib.logger.info(mf,"CEPA0 Iterations have converged!")
        break
    if (it == maxiter):
        raise Exception("Maximum number of iterations exceeded.")
 
lib.logger.info(mf,'CEPA0 Correlation Energy: %5.15f' % (e_cepa0))
lib.logger.info(mf,'CEPA0 Total Energy: %5.15f' % (e_cepa0 + ehf))

if (mol.spin==0):
    eri_aa = ao2mo.kernel(mf._eri, ca, compact=False).reshape([nmoa]*4)
    t2 = t2aa + t2ab + t2ab.transpose(2,3,0,1) + t2bb
    #t2 = 2.0*t2 - t2.transpose(2,3,0,1)
    e1 = numpy.einsum('iajb,aibj', eri_aa[oa,va,oa,va], t2)*0.5
    lib.logger.info(mf,"* Energy with t2 : %5.15f" % e1) 

