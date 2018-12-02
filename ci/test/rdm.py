#!/usr/bin/env python

from pyscf import lib
libhci = lib.load_library('libhci')

def make_rdm12s(civec, norb, nelec):
    '''Spin orbital 1- and 2-particle reduced density matrices (aa, bb, aaaa, aabb, bbbb)
    '''
    strs = civec._strs
    ndet = len(strs)
    rdm1a = numpy.zeros(norb*norb)
    rdm1b = numpy.zeros(norb*norb)
    rdm2aa = numpy.zeros(norb*norb*norb*norb)
    rdm2ab = numpy.zeros(norb*norb*norb*norb)
    rdm2bb = numpy.zeros(norb*norb*norb*norb)

    civec = numpy.asarray(civec, order='C')
    strs = numpy.asarray(strs, order='C')
    rdm1a  = numpy.asarray(rdm1a, order='C')
    rdm1b  = numpy.asarray(rdm1b, order='C')
    rdm2aa = numpy.asarray(rdm2aa, order='C')
    rdm2ab = numpy.asarray(rdm2ab, order='C')
    rdm2bb = numpy.asarray(rdm2bb, order='C')

    # Compute 1- and 2-RDMs
    libhci.compute_rdm12s(ctypes.c_int(norb), 
                          ctypes.c_int(nelec[0]), 
                          ctypes.c_int(nelec[1]), 
                          strs.ctypes.data_as(ctypes.c_void_p), 
                          civec.ctypes.data_as(ctypes.c_void_p), 
                          ctypes.c_ulonglong(ndet), 
                          rdm1a.ctypes.data_as(ctypes.c_void_p),
                          rdm1b.ctypes.data_as(ctypes.c_void_p),
                          rdm2aa.ctypes.data_as(ctypes.c_void_p),
                          rdm2ab.ctypes.data_as(ctypes.c_void_p),
                          rdm2bb.ctypes.data_as(ctypes.c_void_p))

    rdm1a = rdm1a.reshape([norb]*2)
    rdm1b = rdm1b.reshape([norb]*2)
    rdm2aa = rdm2aa.reshape([norb]*4)
    rdm2ab = rdm2ab.reshape([norb]*4)
    rdm2bb = rdm2bb.reshape([norb]*4)

    # Sort 2-RDM into chemists' notation: <p_1 q_2|r_1 s_2> -> (p_1 r_1| q_2 s_2)
    rdm2aa = rdm2aa.transpose(0,2,1,3)
    rdm2ab = rdm2ab.transpose(0,2,1,3)
    rdm2bb = rdm2bb.transpose(0,2,1,3)

    return (rdm1a, rdm1b), (rdm2aa, rdm2ab, rdm2bb)

def make_rdm12(civec, norb, nelec):
    '''Spin orbital 1- and 2-particle reduced density matrices (aa, bb, aaaa, aabb, bbbb)
    '''
    strs = civec._strs
    ndet = len(strs)
    rdm1a = numpy.zeros(norb*norb)
    rdm1b = numpy.zeros(norb*norb)
    rdm2aa = numpy.zeros(norb*norb*norb*norb)
    rdm2ab = numpy.zeros(norb*norb*norb*norb)
    rdm2bb = numpy.zeros(norb*norb*norb*norb)

    civec = numpy.asarray(civec, order='C')
    strs = numpy.asarray(strs, order='C')
    rdm1a  = numpy.asarray(rdm1a, order='C')
    rdm1b  = numpy.asarray(rdm1b, order='C')
    rdm2aa = numpy.asarray(rdm2aa, order='C')
    rdm2ab = numpy.asarray(rdm2ab, order='C')
    rdm2bb = numpy.asarray(rdm2bb, order='C')

    # Compute 1- and 2-RDMs
    libhci.compute_rdm12s(ctypes.c_int(norb), 
                          ctypes.c_int(nelec[0]), 
                          ctypes.c_int(nelec[1]), 
                          strs.ctypes.data_as(ctypes.c_void_p), 
                          civec.ctypes.data_as(ctypes.c_void_p), 
                          ctypes.c_ulonglong(ndet), 
                          rdm1a.ctypes.data_as(ctypes.c_void_p),
                          rdm1b.ctypes.data_as(ctypes.c_void_p),
                          rdm2aa.ctypes.data_as(ctypes.c_void_p),
                          rdm2ab.ctypes.data_as(ctypes.c_void_p),
                          rdm2bb.ctypes.data_as(ctypes.c_void_p))

    rdm1a = rdm1a.reshape([norb]*2)
    rdm1b = rdm1b.reshape([norb]*2)
    rdm2aa = rdm2aa.reshape([norb]*4)
    rdm2ab = rdm2ab.reshape([norb]*4)
    rdm2bb = rdm2bb.reshape([norb]*4)

    # Sort 2-RDM into chemists' notation: <p_1 q_2|r_1 s_2> -> (p_1 r_1| q_2 s_2)
    rdm2aa = rdm2aa.transpose(0,2,1,3)
    rdm2ab = rdm2ab.transpose(0,2,1,3)
    rdm2bb = rdm2bb.transpose(0,2,1,3)

    return rdm1a+rdm1b, rdm2aa+rdm2ab+rdm2ab.transpose(2,3,0,1)+rdm2bb

