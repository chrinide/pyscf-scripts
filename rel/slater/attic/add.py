#!/usr/bin/env python
def add_core_to_rdm(mol, mo_coeff, one_pdm, two_pdm):
    ninact = (mol.nelectron - int(round(numpy.trace(one_pdm))))
    norb = mo_coeff.shape[1]
    nsizerdm = one_pdm.shape[0]
    one_pdm_ = numpy.zeros( (norb, norb) )
    for i in range(ninact):
        one_pdm_[i,i] = 1.0
    one_pdm_[ninact:ninact+nsizerdm,ninact:ninact+nsizerdm] = one_pdm[:,:]
    two_pdm_ = numpy.zeros( (norb, norb, norb, norb) )
    for i in range(ninact):
        for j in range(ninact):
            two_pdm_[i,i,j,j] += 1.0
            two_pdm_[i,j,j,i] -= 1.0
    for p in range(ninact):
        for r in range(ninact,ninact+nsizerdm):
            for s in range(ninact,ninact+nsizerdm):
                two_pdm_[p,p,r,s] += one_pdm_[r,s]
                two_pdm_[r,s,p,p] += one_pdm_[r,s]
                two_pdm_[p,r,s,p] -= one_pdm_[r,s]
                two_pdm_[r,p,p,s] -= one_pdm_[r,s]
    two_pdm_[ninact:ninact+nsizerdm,ninact:ninact+nsizerdm, \
             ninact:ninact+nsizerdm,ninact:ninact+nsizerdm] = two_pdm[:,:]
    return one_pdm_, two_pdm_
