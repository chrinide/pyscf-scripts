#!/usr/bin/env python

import numpy
from pyscf import lib

def spin2spatial(tx, orbspin):
    '''call orbspin_of_sorted_mo_energy to get orbspin'''
    nocc, nvir = tx.shape[1:3]

    idxoa = numpy.where(orbspin[:nocc] == 0)[0]
    idxob = numpy.where(orbspin[:nocc] == 1)[0]
    idxva = numpy.where(orbspin[nocc:] == 0)[0]
    idxvb = numpy.where(orbspin[nocc:] == 1)[0]
    nocc_a = len(idxoa)
    nocc_b = len(idxob)
    nvir_a = len(idxva)
    nvir_b = len(idxvb)

    idxoaa = idxoa[:,None] * nocc + idxoa
    idxoab = idxoa[:,None] * nocc + idxob
    idxobb = idxob[:,None] * nocc + idxob
    idxvaa = idxva[:,None] * nvir + idxva
    idxvab = idxva[:,None] * nvir + idxvb
    idxvbb = idxvb[:,None] * nvir + idxvb
    t2 = tx.reshape(nocc**2,nvir**2)
    t2aa = lib.take_2d(t2, idxoaa.ravel(), idxvaa.ravel())
    t2bb = lib.take_2d(t2, idxobb.ravel(), idxvbb.ravel())
    t2ab = lib.take_2d(t2, idxoab.ravel(), idxvab.ravel())
    t2aa = t2aa.reshape(nocc_a,nocc_a,nvir_a,nvir_a)
    t2bb = t2bb.reshape(nocc_b,nocc_b,nvir_b,nvir_b)
    t2ab = t2ab.reshape(nocc_a,nocc_b,nvir_a,nvir_b)

    return t2aa,t2ab,t2bb

