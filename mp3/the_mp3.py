#!/usr/bin/env python

import numpy
from pyscf import gto, scf, ao2mo, lib
einsum = lib.einsum

# NOT YET WORKING

mol = gto.Mole()
mol.symmetry = 1
mol.atom = '''
O          0.000000000000     0.000000000000    -0.065638538099
H          0.000000000000    -0.757480611647     0.520865616174
H          0.000000000000     0.757480611647     0.520865616174
           '''
mol.basis = 'aug-cc-pvdz'
mol.verbose = 4
mol.build()

mf = scf.RHF(mol)
ehf = mf.kernel()

nao, nmo = mf.mo_coeff.shape
nocc = mol.nelectron/2
nvir = nmo - nocc
c = mf.mo_coeff
co = mf.mo_coeff[:,:nocc]
cv = mf.mo_coeff[:,nocc:]
eo = mf.mo_energy[:nocc]
ev = mf.mo_energy[nocc:]
epsilon = 1.0/(eo.reshape(-1,1,1,1) - ev.reshape(-1,1,1) + eo.reshape(-1,1)- ev)

eri_mo = ao2mo.general(mf._eri, (c,c,c,c), compact=False)
eri_mo = eri_mo.reshape(nmo,nmo,nmo,nmo)
o = slice(0, nocc)
v = slice(nocc, None)

t2 = numpy.zeros((nocc,nvir,nocc,nvir))
t2 = 2.0*einsum('iajb,iajb->iajb', eri_mo[o,v,o,v], epsilon)
t2 -= einsum('ibja,iajb->iajb', eri_mo[o,v,o,v], epsilon)
e_mp2 = numpy.einsum('iajb,iajb->', eri_mo[o,v,o,v], t2)
lib.logger.info(mf,"!*** E(MP2): %12.8f" % e_mp2)
lib.logger.info(mf,"!**** E(HF+MP2): %12.8f" % (e_mp2+ehf))

# MP3 Correlation energy
lib.logger.info(mf,"Starting MP3 energy")
tmpa = eri_mo[o,v,o,v]
tampa = 2.0*tmpa - tmpa.transpose(0,3,2,1)
tampa = numpy.einsum('icjd,icjd->icjd',tampa,epsilon)
tvamp = numpy.einsum('icjd,cadb->iajb',tampa,eri_mo[v,v,v,v])
mp3corr_e_p = numpy.einsum('iajb,iajb->iajb',tvamp,epsilon)
e_p = numpy.einsum('iajb,iajb->', eri_mo[o,v,o,v], mp3corr_e_p)
print e_p

tmpa = eri_mo[o,o,o,o]
tampa = 2.0*tmpa - tmpa.transpose(0,3,2,1)
tvamp = numpy.einsum('kalb,kalb->kalb',eri_mo[o,v,o,v],epsilon)
tvamp = numpy.einsum('kalb,ikjl->iajb',tvamp,tampa)
mp3corr_e_h = numpy.einsum('iajb,iajb->iajb',tvamp,epsilon)
e_h = numpy.einsum('iajb,iajb->', eri_mo[o,v,o,v], mp3corr_e_h)
print e_h

mp3corr_e_i = 0.0
tmpa = eri_mo[o,v,o,v]
tampa = 2.0*tmpa - tmpa.transpose(0,3,2,1)
tampa = numpy.einsum('kajc,kajc->kajc',tampa,epsilon)
tvamp = numpy.einsum('kajc,ikcb->iajb',tampa,eri_mo[o,o,v,v])
mp3corr_e_i -= 2.0*numpy.einsum('iajb,iajb->iajb',tvamp,epsilon)

tmpa = eri_mo[o,v,o,v]
tampa = 2.0*tmpa - tmpa.transpose(0,3,2,1)
tampa = numpy.einsum('jakc,kajc->jakc',tampa,epsilon)
tvamp = numpy.einsum('jakc,ckib->iajb',tampa,eri_mo[v,o,o,v])
mp3corr_e_i -= 2.0*numpy.einsum('iajb,iajb->iajb',tvamp,epsilon)

tmpa = eri_mo[o,v,o,v]
tampa = 2.0*tmpa - tmpa.transpose(0,3,2,1)
tampa = numpy.einsum('jakc,kajc->jakc',tampa,epsilon)
tvamp = numpy.einsum('jakc,ikcb->iajb',tampa,eri_mo[o,o,v,v])
mp3corr_e_i -= 2.0*numpy.einsum('iajb,iajb->iajb',tvamp,epsilon)

tmpa = eri_mo[o,v,o,v]
tampa = 2.0*tmpa - 4.0*tmpa.transpose(0,3,2,1)
tampa = numpy.einsum('kajc,kajc->kajc',tampa,epsilon)
tvamp = numpy.einsum('kajc,ckib->iajb',tampa,eri_mo[v,o,o,v])
mp3corr_e_i -= 2.0*numpy.einsum('iajb,iajb->iajb',tvamp,epsilon)
e_i = numpy.einsum('iajb,iajb->', eri_mo[o,v,o,v], mp3corr_e_i)
print e_i

t2 = mp3corr_e_p + mp3corr_e_h + mp3corr_e_i
e_mp3 = numpy.einsum('iajb,iajb->', eri_mo[o,v,o,v], t2)
lib.logger.info(mf,"!*** E(MP3): %12.8f" % e_mp3)
e_mp3 += e_mp2
lib.logger.info(mf,"!*** E(MP2+MP3): %12.8f" % e_mp3)
mp3 = ehf + e_mp3
lib.logger.info(mf,"!**** E(HF+MP2+MP3): %12.8f" % (e_mp3+ehf))

