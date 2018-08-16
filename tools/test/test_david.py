#!/usr/bin/env python

import numpy
from pyscf import lib

n = 100
a = numpy.random.rand(n,n)
a = a + a.T
def matvec(x):
    return a.dot(x)
precond = a.diagonal()
x_init = numpy.zeros(n)
x_init[0] = 1
#e, c = lib.eigh(matvec, x_init, precond, nroots=4, max_cycle=1000, verbose=5)
e, c = lib.eig(matvec, x_init, precond, nroots=4, max_cycle=1000, verbose=5)
print('Eigenvalues', e)
