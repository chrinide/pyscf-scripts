#!/usr/bin/env python

import numpy
from pyscf import gto, scf, dft, tddft, mp, cc

mol = gto.Mole()
mol.atom = [
    ['H' , (0. , 0. , .917)],
    ['F' , (0. , 0. , 0.)], ]
mol.basis = 'aug-cc-pvtz'
mol.build()

def diagonalize(a, b):
    nocc, nvir = a.shape[:2]
    aa = a.reshape(nocc*nvir,nocc*nvir)
    bb = b.reshape(nocc*nvir,nocc*nvir)
    e = numpy.linalg.eig(numpy.bmat([[ aa       , bb      ],
                                     [-bb.conj(),-aa.conj()]]))[0]
    e = numpy.sort(e[e > 0])
    return e

nroots = 5
mf = scf.RHF(mol).run()
a, b = tddft.TDHF(mf).get_ab()
e = diagonalize(a,b)
print('TDHF:', e[:nroots])
nocc, nvir = a.shape[:2]
a = a.reshape(nocc*nvir,nocc*nvir)

etda = numpy.linalg.eig(a)[0]
etda = numpy.sort(etda)
print('TDA:', etda[:nroots])

e_corr = 0.5*sum(e-etda)
print('RPA correlation energy', e_corr)

pt2 = mp.MP2(mf)
pt2.kernel()

ccsd = cc.CCSD(mf)
ccsd.kernel()
old_update_amps = ccsd.update_amps
def update_amps(t1, t2, eris):
    t1, t2 = old_update_amps(t1, t2, eris)
    return t1*0, t2
ccsd.update_amps = update_amps
ccsd.verbose = 0
ccsd.kernel()
print('CCD correlation energy', ccsd.e_corr)

