#!/usr/bin/env python

import time, os
import numpy as np
from pyscf import gto, scf, ao2mo

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
SCF_E = mf.kernel()

nao, nmo = mf.mo_coeff.shape
nocc = mol.nelectron/2
nvir = nmo - nocc
c = mf.mo_coeff
co = mf.mo_coeff[:,:nocc]
cv = mf.mo_coeff[:,nocc:]
eo = mf.mo_energy[:nocc]
ev = mf.mo_energy[nocc:]

MO = ao2mo.general(mf._eri, (c,c,c,c), compact=False)
MO = MO.reshape(nmo,nmo,nmo,nmo)

# Build epsilon tensor
eocc = mf.mo_energy[:nocc]
evirt = mf.mo_energy[nocc:]
epsilon = 1/(eocc.reshape(-1, 1, 1, 1) + eocc.reshape(-1, 1, 1) - evirt.reshape(-1, 1) - evirt)

# Build o and v slices
o = slice(0, nocc)
v = slice(nocc, MO.shape[0])

# To Szabo physics notations
MO = MO.swapaxes(1, 2)

# Particle denisty matrix
rdm2 = np.zeros((nocc,nvir,nocc,nvir))

### MP2 correlation energy
MP2corr_E = 2 * np.einsum('rsab,abrs->abrs', MO[v, v, o, o], epsilon)
MP2corr_E -= np.einsum('rsba,abrs->abrs', MO[v, v, o, o], epsilon)

rdm2 = MP2corr_E.swapaxes(1,2)*2
MP2corr_E = np.einsum('iajb,iajb->', MO[o,o,v,v].swapaxes(1,2), MP2corr_E.swapaxes(1,2))
MP2total_E = SCF_E + MP2corr_E
print('MP2 correlation energy: %16.8f' % MP2corr_E)
print('MP2 total energy:       %16.8f' % MP2total_E)

# MP3 Correlation energy
# Prefactors taken from terms in unnumbered expression for spatial-orbital MP3
# energy on [Szabo:1996] pp. (bottom) 367 - (top) 368. Individual equations taken
# from [Szabo:1996] Tbl. 6.2 pp. 364-365
print('\nStarting MP3 energy...')
t = time.time()

# Equation 1: 3rd order diagram 1
MP3corr_E =   2.0 * np.einsum('ruts,tsab,abru,abts->abru', MO[v, v, v, v], MO[v, v, o, o], epsilon, epsilon) 
# Equation 2: 3rd order diagram 2 
MP3corr_E +=  2.0 * np.einsum('cbad,rscb,adrs,cbrs->adrs', MO[o, o, o, o], MO[v, v, o, o], epsilon, epsilon)
# Equation 3: 3rd order diagram 3
MP3corr_E += -4.0 * np.einsum('rbsc,stab,acrt,abst->acrt', MO[v, o, v, o], MO[v, v, o, o], epsilon, epsilon)
# Equation 4: 3rd order diagram 4
MP3corr_E += -4.0 * np.einsum('rasb,stac,bcrt,acst->bcrt', MO[v, o, v, o], MO[v, v, o, o], epsilon, epsilon)
# Equation 5: 3rd order diagram 5
MP3corr_E +=  8.0 * np.einsum('btsc,rsab,acrt,abrs->acrt', MO[o, v, v, o], MO[v, v, o, o], epsilon, epsilon)
# Equation 6: 3rd order diagram 6
MP3corr_E +=  2.0 * np.einsum('atsc,rsab,cbrt,abrs->cbrt', MO[o, v, v, o], MO[v, v, o, o], epsilon, epsilon)
# Equation 7: 3rd order diagram 7
MP3corr_E += -1.0 * np.einsum('dbac,srdb,acrs,dbrs->acrs', MO[o, o, o, o], MO[v, v, o, o], epsilon, epsilon)
# Equation 8: 3rd order diagram 8
MP3corr_E += -1.0 * np.einsum('trus,usab,abtr,abus->abrt', MO[v, v, v, v], MO[v, v, o, o], epsilon, epsilon)
# Equation 9: 3rd order diagram 9
MP3corr_E +=  2.0 * np.einsum('arbs,tsac,cbrt,acst->bcrt', MO[o, v, o, v], MO[v, v, o, o], epsilon, epsilon)
# Equation 10: 3rd order diagram 10
MP3corr_E +=  2.0 * np.einsum('rasb,stac,cbrt,acst->cbrt', MO[v, o, v, o], MO[v, v, o, o], epsilon, epsilon)
# Equation 11: 3rd order diagram 11
MP3corr_E += -4.0 * np.einsum('scat,rtbc,abrs,cbrt->abrs', MO[v, o, o, v], MO[v, v, o, o], epsilon, epsilon)
# Equation 12: 3rd order diagram 12
MP3corr_E += -4.0 * np.einsum('atsc,rsab,bctr,abrs->bcrt', MO[o, v, v, o], MO[v, v, o, o], epsilon, epsilon)

rdm2 += MP3corr_E.swapaxes(1,2)*2
MP3corr_E = np.einsum('iajb,iajb->', MO[o,o,v,v].swapaxes(1,2), MP3corr_E.swapaxes(1,2))
print('Third order energy:     %16.8f' % MP3corr_E)
print('...took %.3f seconds to compute MP3 correlation energy.' % (time.time()-t))
MP3corr_E += MP2corr_E
MP3total_E = SCF_E + MP3corr_E
print('MP3 correlation energy: %16.8f' % MP3corr_E)
print('MP3 total energy:       %16.8f' % MP3total_E)

