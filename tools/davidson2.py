#!/usr/bin/env python

from __future__ import division
import math
import numpy as np
import time

''' Block Davidson, Joshua Goings (2013)

    Block Davidson method for finding the first few
        lowest eigenvalues of a large, diagonally dominant,
    sparse Hermitian matrix (e.g. Hamiltonian)
'''

n = 1200                                        # Dimension of matrix
tol = 1e-8                              # Convergence tolerance
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

########################################################

# Build a fake sparse symmetric matrix 
n = 1600
print('Dimension of the matrix',n,'*',n)
sparsity = 0.001
A = np.zeros((n,n))
for i in range(0,n) : 
    A[i,i] = i-9
A = A + sparsity*np.random.randn(n,n)
A = (A.T + A)/2

tol = 1e-9             # Convergence tolerance
mmax = 20              # Maximum number of iterations

# Setup the subspace trial vectors
k = 4
print ('No. of start vectors:',k)
neig = 3
print ('No. of desired Eigenvalues:',neig)
t = np.eye(n,k) # initial trial vectors
v = np.zeros((n,n)) # holder for trial vectors as iterations progress
I = np.eye(n) # n*n identity matrix
ritz = np.zeros((n,n))
f = np.zeros((n,n))
#-------------------------------------------------------------------------------
# Begin iterations  
#-------------------------------------------------------------------------------
start = time.time()
iter = 0
for m in range(k,mmax,k):
    iter = iter + 1
    print ("Iteration no:", iter)
    if iter==1:  # for first iteration add normalized guess vectors to matrix v
        for l in range(m):
            v[:,l] = t[:,l]/(np.linalg.norm(t[:,l]))
    # Matrix-vector products, form the projected Hamiltonian in the subspace
    T = np.linalg.multi_dot([v[:,:m].T,A,v[:,:m]]) # selects fastest evaluation order
    w, vects = np.linalg.eig(T) # Diagonalize the subspace Hamiltonian
    jj = 0
    s = w.argsort()
    ss = w[s]
    #***************************************************************************
    # For each eigenvector of T build a Ritz vector, precondition it and check
    # if the norm is greater than a set threshold.
    #***************************************************************************
    for ii in range(m): #for each new eigenvector of T
        f = np.diag(1./ np.diag((np.diag(np.diag(A)) - w[ii]*I)))
#        print (f)
        ritz[:,ii] = np.dot(f,np.linalg.multi_dot([(A-w[ii]*I),v[:,:m],vects[:,ii]]))
        if np.linalg.norm(ritz[:,ii]) > 1e-7 :
            ritz[:,ii] = ritz[:,ii]/(np.linalg.norm(ritz[:,ii]))
            v[:,m+jj] = ritz[:,ii]
            jj = jj + 1
    q, r = np.linalg.qr(v[:,:m+jj-1])
    for kk in range(m+jj-1):
        v[:,kk] = q[:,kk]
    for ii in range(neig):
        print (ss[ii])
    if iter==1: 
        check_old = ss[:neig]
        check_new = 1
    elif iter==2:
        check_new = ss[:neig]
    else: 
        check_old = check_new
        check_new = ss[:neig]
    check = np.linalg.norm(check_new - check_old)
    if check < tol:
        print('Block Davidson converged at iteration no.:',iter)
        break
end = time.time()
print ('Block Davidson time:',end-start)
start = time.time()
eig, eigvecs = np.linalg.eig(A)
end = time.time() 
s = eig.argsort()
ss = eig[s]
print('Exact Diagonalization:')
for ii in range(neig):    
    print(ss[ii])
#print(ss[:neig])
print ('Exact Diagonalization time:',end-start)
     
