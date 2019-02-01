#!/usr/bin/env python

from __future__ import division
import math
import numpy as np
import time

np.random.seed(4)

''' Block Davidson, Joshua Goings (2013)

    Block Davidson method for finding the first few
        lowest eigenvalues of a large, diagonally dominant,
    sparse Hermitian matrix (e.g. Hamiltonian)
'''

n = 4000                                        # Dimension of matrix
tol = 1e-6                              # Convergence tolerance
mmax = n//2                             # Maximum number of iterations  

''' Create sparse, diagonally dominant matrix A with 
        diagonal containing 1,2,3,...n. The eigenvalues
    should be very close to these values. You can 
    change the sparsity. A smaller number for sparsity
    increases the diagonal dominance. Larger values
    (e.g. sparsity = 1) create a dense matrix
'''

sparsity = 0.0001
A = np.zeros((n,n))
for i in range(0,n):
    A[i,i] = i + 1 
A = A + sparsity*np.random.randn(n,n) 
A = (A.T + A)/2 


k = 8                                   # number of initial guess vectors 
eig = 4                                 # number of eignvalues to solve 
t = np.eye(n,k)                 # set of k unit vectors as guess
V = np.zeros((n,n))             # array of zeros to hold guess vec
I = np.eye(n)                   # identity matrix same dimen as A

# Begin block Davidson routine

start_davidson = time.time()

for m in xrange(k,mmax,k):
    if m <= k:
        for j in xrange(0,k):
            V[:,j] = t[:,j]/np.linalg.norm(t[:,j])
        theta_old = 1 
    elif m > k:
        theta_old = theta[:eig]
    V,R = np.linalg.qr(V)
    T = np.dot(V[:,:(m+1)].T,np.dot(A,V[:,:(m+1)]))
    THETA,S = np.linalg.eig(T)
    idx = THETA.argsort()
    theta = THETA[idx]
    s = S[:,idx]
    for j in xrange(0,k):
        w = np.dot((A - theta[j]*I),np.dot(V[:,:(m+1)],s[:,j])) 
#        q = np.dot(np.linalg.inv(theta[j]*I - np.diag(np.diag(A))),w)
        q = w/(theta[j]-A[j,j])
        V[:,(m+j+1)] = q
    norm = np.linalg.norm(theta[:eig] - theta_old)
    if norm < tol:
        break

end_davidson = time.time()

# End of block Davidson. Print results.

print "davidson = ", theta[:eig],";",\
    end_davidson - start_davidson, "seconds"

# Begin Numpy diagonalization of A

start_numpy = time.time()

E,Vec = np.linalg.eig(A)
E = np.sort(E)

end_numpy = time.time()

# End of Numpy diagonalization. Print results.

print "numpy = ", E[:eig],";",\
     end_numpy - start_numpy, "seconds" 
