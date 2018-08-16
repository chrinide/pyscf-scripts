#!/usr/bin/env python

import numpy

def appendColumn(A,newVec):
    """  
    Append a column vector onto matrix A; this creates a new
    matrix and does the relevant copying.
    """
    n,m = A.shape
    Anew = numpy.zeros((n,m+1),'d')
    Anew[:,:m] = A
    Anew[:,m] = newVec
    return Anew

def orthog(q,qs,**kwargs):
    "Orthonormalize vector q to set of vectors qs"
    nbasis,nvec = qs.shape
    north = kwargs.get('north',nvec)
    for i in range(north):
        olap = numpy.dot(q,qs[:,i])
        q -= olap*qs[:,i]
    norm = numpy.sqrt(numpy.dot(q,q))
    return norm

def davidson(A,nroots,**kwargs):
    etol = kwargs.get('etol',1e-6) # tolerance on the eigenval convergence
    ntol = kwargs.get('ntol',1e-10) # tolerance on the vector norms for addn
    n,m = A.shape
    ninit = max(nroots,2)
    B = numpy.zeros((n,ninit),'d')
    for i in range(ninit): B[i,i] = 1.

    nc = 0 # number of converged roots
    eigold = 1e10
    for iter in range(n):
        if nc >= nroots: break
        D = numpy.dot(A,B)
        S = numpy.dot(B.T,D)
        m = len(S)
        eval,evec = numpy.linalg.eigh(S)

        bnew = numpy.zeros(n,'d')
        for i in range(m):
            bnew += evec[i,nc]*(D[:,i] - eval[nc]*B[:,i])

        for i in range(n):
            denom = max(eval[nc]-A[i,i],1e-8) # Maximum amplification factor
            bnew[i] /= denom

        norm = orthog(bnew,B)
        bnew = bnew / norm

        if abs(eval[nc]-eigold) < etol:
            nc += 1
        eigold = eval[nc]
        if norm > ntol: B = appendColumn(B,bnew)

    E = eval[:nroots]
    nv = len(evec)
    V = numpy.dot(B[:,:nv],evec)
    return E,V

if __name__ == '__main__': 
        nroots = 5
        n = 100
        a = numpy.random.rand(n,n)
        a = a + a.T
        e,v = davidson(a,nroots) 
        print "Davidson", e
        eval, evec = numpy.linalg.eigh(a)
        print "Exact", eval[:nroots]
