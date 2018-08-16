#!/usr/bin/env python

import numpy

def symorth(s):
    "symmetric orthogonalization"
    e,u = numpy.linalg.eigh(s)
    n = len(e)
    shalf = numpy.identity(n,'d')
    for i in range(n):
        shalf[i,i] /= numpy.sqrt(e[i])
    return simx(shalf,u,True)

def canorth(s):
    "canonical orthogonalization u/sqrt(lambda)"
    e,u = numpy.linalg.eigh(s)
    for i in range(len(e)):
        u[:,i] = u[:,i] / numpy.sqrt(e[i])
    return u

def cholorth(s):
    "cholesky orthogonalization"
    return numpy.linalg.inv(numpy.linalg.cholesky(s)).T

def simx(a,b,transpose=False):
    "similarity transform b^t(ab) or b(ab^t) (if transpose)"
    if transpose:
        return numpy.dot(b,numpy.dot(a,b.T))
    return numpy.dot(b.T,numpy.dot(a,b))

def geigh(h,s):
    "solve the generalized eigensystem hc = esc"
    a = cholorth(s)
    e,u = numpy.linalg.eigh(simx(h,a))
    return e,numpy.dot(a,u)


def ao2mo(h,c): return simx(h,c)
def mo2ao(h,c,s): return simx(h,numpy.dot(s,c),transpose=True)

