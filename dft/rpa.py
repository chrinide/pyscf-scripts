#!/usr/bin/env python

import numpy
from pyscf import gto, scf, dft, tddft, mp, cc

mol = gto.Mole()
mol.atom = [
    ['N' , (0. , 0. , 1.1)],
    ['N' , (0. , 0. , 0.0)], ]
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

mf = dft.RKS(mol)
mf.xc = 'pbe,pbe'
mf.kernel()
a, b = tddft.dRPA(mf).get_ab()
e = diagonalize(a,b)
print('dTD:', e[:nroots])
nocc, nvir = a.shape[:2]
a = a.reshape(nocc*nvir,nocc*nvir)

etda = numpy.linalg.eig(a)[0]
etda = numpy.sort(etda)
print('dTDA:', etda[:nroots])

e_corr = 0.5*sum(e-etda)
print('dRPA correlation energy', e_corr)
