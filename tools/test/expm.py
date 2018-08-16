#!/usr/bin/env python

import numpy

def expm(a,tol=1e-6,maxit=40):
    """
    http://epubs.siam.org/doi/abs/10.1137/S00361445024180
    """
    n,m = a.shape
    assert(n==m)
    factor = 1.0
    x =  numpy.identity(n,'d')
    em = numpy.identity(n,'d')
    for j in range(1,maxit):
        factor /= j
        x = numpy.dot(x,a)
        if numpy.linalg.norm(x) < tol:
            break
        em += factor*x
    else:
        print("expm remainder = \n%s" % x)
        raise Exception("Maximum iterations reached in expm")
    return em

