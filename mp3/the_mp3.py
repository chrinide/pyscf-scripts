#!/usr/bin/env python

import time, os
import numpy as np
from pyscf import gto, scf, ao2mo, lib
einsum = lib.einsum

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

# Hartree-Fock
mf = scf.RHF(mol)
SCF_E = mf.kernel()

# Get info
nao, nmo = mf.mo_coeff.shape
nocc = mol.nelectron/2
nvir = nmo - nocc
c = mf.mo_coeff
co = mf.mo_coeff[:,:nocc]
cv = mf.mo_coeff[:,nocc:]
eo = mf.mo_energy[:nocc]
ev = mf.mo_energy[nocc:]

# Integral generation
MO = ao2mo.general(mf._eri, (c,c,c,c), compact=False)
MO = MO.reshape(nmo,nmo,nmo,nmo)

# Build epsilon tensor
eocc = mf.mo_energy[:nocc]
evirt = mf.mo_energy[nocc:]
#epsilon = 1.0/(eocc.reshape(-1, 1, 1, 1) + eocc.reshape(-1, 1, 1) - evirt.reshape(-1, 1) - evirt)
epsilon = 1.0 / (eo.reshape(-1, 1, 1, 1) - ev.reshape(-1, 1, 1) + eo.reshape(-1, 1) - ev)

# Build o and v slices
o = slice(0, nocc)
v = slice(nocc, MO.shape[0])

### MP2 correlation energy
MP2corr_E = 2.0 * np.einsum('iajb,iajb->iajb', MO[o,v,o,v], epsilon)
MP2corr_E -= np.einsum('ibja,iajb->iajb', MO[o,v,o,v], epsilon)
MP2corr_E = np.einsum('iajb,iajb->', MO[o,v,o,v], MP2corr_E)
MP2total_E = SCF_E + MP2corr_E
print('MP2 correlation energy: %16.8f' % MP2corr_E)
print('MP2 total energy:       %16.8f' % MP2total_E)

# MP3 Correlation energy
print('Starting MP3 energy...')
tmpa = MO[o,v,o,v]
tampa = 2.0*tmpa - tmpa.transpose(0,3,2,1)
tampa = np.einsum('icjd,icjd->icjd',tampa,epsilon)
tvamp = np.einsum('icjd,cadb->iajb',tampa,MO[v,v,v,v])
MP3corr_E_p = np.einsum('iajb,iajb->iajb',tvamp,epsilon)
e_p = np.einsum('iajb,iajb->', MO[o,v,o,v], MP3corr_E_p)
print e_p

tmpa = MO[o,o,o,o]
tampa = 2.0*tmpa - tmpa.transpose(0,3,2,1)
tvamp = np.einsum('kalb,kalb->kalb',MO[o,v,o,v],epsilon)
tvamp = np.einsum('kalb,ikjl->iajb',tvamp,tampa)
MP3corr_E_h = np.einsum('iajb,iajb->iajb',tvamp,epsilon)
e_h = np.einsum('iajb,iajb->', MO[o,v,o,v], MP3corr_E_h)
print e_h

MP3corr_E_i = 0.0
tmpa = MO[o,v,o,v]
tampa = 2.0*tmpa - tmpa.transpose(0,3,2,1)
tampa = np.einsum('kajc,kajc->kajc',tampa,epsilon)
tvamp = np.einsum('kajc,ikcb->iajb',tampa,MO[o,o,v,v])
MP3corr_E_i -= 2.0*np.einsum('iajb,iajb->iajb',tvamp,epsilon)

tmpa = MO[o,v,o,v]
tampa = 2.0*tmpa - tmpa.transpose(0,3,2,1)
tampa = np.einsum('jakc,kajc->jakc',tampa,epsilon)
tvamp = np.einsum('jakc,ckib->iajb',tampa,MO[v,o,o,v])
MP3corr_E_i -= 2.0*np.einsum('iajb,iajb->iajb',tvamp,epsilon)

tmpa = MO[o,v,o,v]
tampa = 2.0*tmpa - tmpa.transpose(0,3,2,1)
tampa = np.einsum('jakc,kajc->jakc',tampa,epsilon)
tvamp = np.einsum('jakc,ikcb->iajb',tampa,MO[o,o,v,v])
MP3corr_E_i -= 2.0*np.einsum('iajb,iajb->iajb',tvamp,epsilon)

tmpa = MO[o,v,o,v]
tampa = 2.0*tmpa - 4.0*tmpa.transpose(0,3,2,1)
tampa = np.einsum('kajc,kajc->kajc',tampa,epsilon)
tvamp = np.einsum('kajc,ckib->iajb',tampa,MO[v,o,o,v])
MP3corr_E_i -= 2.0*np.einsum('iajb,iajb->iajb',tvamp,epsilon)
e_i = np.einsum('iajb,iajb->', MO[o,v,o,v], MP3corr_E_i)
print e_i

MP3corr_E = MP3corr_E_p + MP3corr_E_h + MP3corr_E_i
MP3corr_E = np.einsum('iajb,iajb->', MO[o,v,o,v], MP3corr_E)
print('Third order energy:     %16.8f' % MP3corr_E)
MP3corr_E += MP2corr_E
MP3total_E = SCF_E + MP3corr_E
print('MP3 correlation energy: %16.8f' % MP3corr_E)
print('MP3 total energy:       %16.8f' % MP3total_E)
