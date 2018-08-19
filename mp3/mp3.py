#!/usr/bin/env python

import time, os, sys, numpy
from pyscf import gto, scf, ao2mo, lib
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
epsilon = 1.0/(eo.reshape(-1,1,1,1) - ev.reshape(-1,1,1) + eo.reshape(-1,1) - ev)
o = slice(0, nocc)
v = slice(nocc, None)

t2 = numpy.zeros((nocc,nvir,nocc,nvir))
t2 = 2.0*numpy.einsum('aibj,iajb->iajb', eri_mo[v,o,v,o], epsilon)
t2 -= numpy.einsum('biaj,iajb->iajb', eri_mo[v,o,v,o], epsilon)
e_mp2 = numpy.einsum('aibj,iajb->', eri_mo[v,o,v,o], t2)
lib.logger.info(mf,"!*** E(MP2): %12.8f" % e_mp2)
lib.logger.info(mf,"!**** E(HF+MP2): %12.8f" % (e_mp2+ehf))

# MP3 Correlation energy
# Prefactors taken from terms in unnumbered expression for spatial-orbital MP3
# energy on [Szabo:1996] pp. (bottom) 367 - (top) 368. Individual equations taken
# from [Szabo:1996] Tbl. 6.2 pp. 364-365
t = time.time()
lib.logger.info(mf,"Starting MP3 energy")
t2 = numpy.zeros((nocc,nvir,nocc,nvir))
factor = numpy.einsum('aibj,iajb->aibj',eri_mo[v,o,v,o],epsilon)
# Equation 1: 3rd order diagram 1
t2 = 2.0*numpy.einsum('rtus,tasb,arbu->arbu', eri_mo[v,v,v,v], factor, epsilon) 
# Equation 2: 3rd order diagram 2 
t2 += 2.0*numpy.einsum('cabd,rcsb,ards->ards', eri_mo[o,o,o,o], factor, epsilon)
# Equation 3: 3rd order diagram 3
t2 += -4.0*numpy.einsum('rsbc,satb,arct->arct', eri_mo[v,v,o,o], factor, epsilon)
# Equation 4: 3rd order diagram 4
t2 += -4.0*numpy.einsum('rsab,satc,brct->brct', eri_mo[v,v,o,o], factor, epsilon)
# Equation 5: 3rd order diagram 5
t2 += 8.0*numpy.einsum('bstc,rasb,arct->arct', eri_mo[o,v,v,o], factor, epsilon)
# Equation 6: 3rd order diagram 6
t2 += 2.0*numpy.einsum('astc,rasb,crbt->crbt', eri_mo[o,v,v,o], factor, epsilon)
# Equation 7: 3rd order diagram 7
t2 += -1.0*numpy.einsum('dabc,sdrb,arcs->arcs', eri_mo[o,o,o,o], factor, epsilon)
# Equation 8: 3rd order diagram 8
t2 += -1.0*numpy.einsum('turs,uasb,atbr->arbt', eri_mo[v,v,v,v], factor, epsilon)
# Equation 9: 3rd order diagram 9
t2 += 2.0*numpy.einsum('abrs,tasc,crbt->brct', eri_mo[o,o,v,v], factor, epsilon)
# Equation 10: 3rd order diagram 10
t2 += 2.0*numpy.einsum('rsab,satc,crbt->crbt', eri_mo[v,v,o,o], factor, epsilon)
# Equation 11: 3rd order diagram 11
t2 += -4.0*numpy.einsum('sact,rbtc,arbs->arbs', eri_mo[v,o,o,v], factor, epsilon)
# Equation 12: 3rd order diagram 12
t2 += -4.0*numpy.einsum('astc,rasb,btcr->brct', eri_mo[o,v,v,o], factor, epsilon)
e_mp3 = numpy.einsum('aibj,iajb->', eri_mo[v,o,v,o], t2)
lib.logger.info(mf,"!*** E(MP3): %12.8f" % e_mp3)
lib.logger.info(mf,"!*** E(MP2.5): %12.8f" % (0.5*e_mp3+e_mp2))
lib.logger.info(mf,"!*** E(MP2+MP3): %12.8f" % (e_mp2+e_mp3))
lib.logger.info(mf,'...took %.3f seconds' % (time.time()-t))

