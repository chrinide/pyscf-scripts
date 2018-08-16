#!/usr/bin/env python

import time, os, sys, numpy
from pyscf import gto, scf, ao2mo, lib
from pyscf.tools import molden
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

mf = scf.RHF(mol)
ehf = mf.kernel()

nao, nmo = mf.mo_coeff.shape
ncore = 0
nocc = mol.nelectron/2 - ncore
nvir = nmo - nocc - ncore
c = mf.mo_coeff[:,ncore:ncore+nocc+nvir]
eo = mf.mo_energy[ncore:ncore+nocc]
ev = mf.mo_energy[ncore+nocc:]

eri_mo = ao2mo.general(mf._eri, (c,c,c,c), compact=False)
eri_mo = eri_mo.reshape(nocc+nvir,nocc+nvir,nocc+nvir,nocc+nvir)
epsilon = 1.0/(eo.reshape(-1,1,1,1) + eo.reshape(-1,1,1) - ev.reshape(-1,1) - ev)
o = slice(0, nocc)
v = slice(nocc, None)
eri_mo = eri_mo.swapaxes(1, 2)

t2 = numpy.zeros((nocc,nvir,nocc,nvir))
t2 = 2.0*numpy.einsum('rsab,abrs->abrs', eri_mo[v,v,o,o], epsilon)
t2 -= numpy.einsum('rsba,abrs->abrs', eri_mo[v,v,o,o], epsilon)
t2 = t2.swapaxes(1,2)
e_mp2 = numpy.einsum('iajb,iajb->', eri_mo[o,o,v,v].swapaxes(1,2), t2)
lib.logger.info(mf,"!*** E(MP2): %12.8f" % e_mp2)
lib.logger.info(mf,"!**** E(HF+MP2): %12.8f" % (e_mp2+ehf))

# MP3 Correlation energy
# Prefactors taken from terms in unnumbered expression for spatial-orbital MP3
# energy on [Szabo:1996] pp. (bottom) 367 - (top) 368. Individual equations taken
# from [Szabo:1996] Tbl. 6.2 pp. 364-365
t = time.time()
lib.logger.info(mf,"Starting MP3 energy")
t2 = numpy.zeros((nocc,nocc,nvir,nvir))
factor = numpy.einsum('abij,ijab->abij',eri_mo[v,v,o,o],epsilon)
# Equation 1: 3rd order diagram 1
t2 = 2.0*numpy.einsum('ruts,tsab,abru->abru', eri_mo[v,v,v,v], factor, epsilon) 
# Equation 2: 3rd order diagram 2 
t2 += 2.0*numpy.einsum('cbad,rscb,adrs->adrs', eri_mo[o,o,o,o], factor, epsilon)
# Equation 3: 3rd order diagram 3
t2 += -4.0*numpy.einsum('rbsc,stab,acrt->acrt', eri_mo[v,o,v,o], factor, epsilon)
# Equation 4: 3rd order diagram 4
t2 += -4.0*numpy.einsum('rasb,stac,bcrt->bcrt', eri_mo[v,o,v,o], factor, epsilon)
# Equation 5: 3rd order diagram 5
t2 += 8.0*numpy.einsum('btsc,rsab,acrt->acrt', eri_mo[o,v,v,o], factor, epsilon)
# Equation 6: 3rd order diagram 6
t2 += 2.0*numpy.einsum('atsc,rsab,cbrt->cbrt', eri_mo[o,v,v,o], factor, epsilon)
# Equation 7: 3rd order diagram 7
t2 += -1.0*numpy.einsum('dbac,srdb,acrs->acrs', eri_mo[o,o,o,o], factor, epsilon)
# Equation 8: 3rd order diagram 8
t2 += -1.0*numpy.einsum('trus,usab,abtr->abrt', eri_mo[v,v,v,v], factor, epsilon)
# Equation 9: 3rd order diagram 9
t2 += 2.0*numpy.einsum('arbs,tsac,cbrt->bcrt', eri_mo[o,v,o,v], factor, epsilon)
# Equation 10: 3rd order diagram 10
t2 += 2.0*numpy.einsum('rasb,stac,cbrt->cbrt', eri_mo[v,o,v,o], factor, epsilon)
# Equation 11: 3rd order diagram 11
t2 += -4.0*numpy.einsum('scat,rtbc,abrs->abrs', eri_mo[v,o,o,v], factor, epsilon)
# Equation 12: 3rd order diagram 12
t2 += -4.0*numpy.einsum('atsc,rsab,bctr->bcrt', eri_mo[o,v,v,o], factor, epsilon)
t2 = t2.swapaxes(1,2)
e_mp3 = numpy.einsum('iajb,iajb->', eri_mo[o,o,v,v].swapaxes(1,2), t2)
lib.logger.info(mf,"!*** E(MP3): %12.8f" % e_mp3)
lib.logger.info(mf,'...took %.3f seconds' % (time.time()-t))

