#!/usr/bin/env python

import numpy

def add_inactive_space_to_rdm(mol, nsize, one_pdm, two_pdm):
    '''This function will add them and return the full density matrices'''

    # Find number of inactive electrons by taking the number of electrons
    # as the trace of the 1RDM, and subtracting from the total number of
    # electrons
    ninact = (mol.nelectron - int(round(numpy.trace(one_pdm))))/2
    norb = nsize
    nsizerdm = one_pdm.shape[0]

    one_pdm_ = numpy.zeros((norb,norb))
    # Add the core first.
    for i in range(ninact):
        one_pdm_[i,i] = 2.0

    # Add the rest of the density matrix.
    one_pdm_[ninact:ninact+nsizerdm,ninact:ninact+nsizerdm] = one_pdm[:,:]

    two_pdm_ = numpy.zeros((norb,norb,norb,norb))
    
    # Add on frozen core contribution, assuming that the inactive orbitals are
    # doubly occupied.
    for i in range(ninact):
        for j in range(ninact):
            two_pdm_[i,i,j,j] +=  4.0
            two_pdm_[i,j,j,i] += -2.0

    # Inactve-Active elements.
    for p in range(ninact):
        for r in range(ninact,ninact+nsizerdm):
            for s in range(ninact,ninact+nsizerdm):
                two_pdm_[p,p,r,s] += 2.0*one_pdm_[r,s]
                two_pdm_[r,s,p,p] += 2.0*one_pdm_[r,s]
                two_pdm_[p,r,s,p] -= one_pdm_[r,s]
                two_pdm_[r,p,p,s] -= one_pdm_[r,s]

    # Add active space.
    two_pdm_[ninact:ninact+nsizerdm,ninact:ninact+nsizerdm, \
             ninact:ninact+nsizerdm,ninact:ninact+nsizerdm] = \
             two_pdm[:,:]

    return one_pdm_, two_pdm_

def add_inactive_space_to_rdm1(mol, nsize, one_pdm):
    '''This function will add them and return the full density matrices'''

    # Find number of inactive electrons by taking the number of electrons
    # as the trace of the 1RDM, and subtracting from the total number of
    # electrons
    ninact = (mol.nelectron - int(round(numpy.trace(one_pdm))))/2
    norb = nsize
    nsizerdm = one_pdm.shape[0]

    one_pdm_ = numpy.zeros((norb,norb))
    # Add the core first.
    for i in range(ninact):
        one_pdm_[i,i] = 2.0

    # Add the rest of the density matrix.
    one_pdm_[ninact:ninact+nsizerdm,ninact:ninact+nsizerdm] = one_pdm[:,:]

    return one_pdm_

