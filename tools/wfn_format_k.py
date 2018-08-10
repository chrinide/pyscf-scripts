#!/usr/bin/env python

from functools import reduce
from pyscf import gto
from pyscf import lib
import numpy

TYPE_MAP = [
    [1],  # S
    [2, 3, 4],  # P
    [5, 8, 9, 6, 10, 7],  # D
    [11,14,15,17,20,18,12,16,19,13],  # F
    [21,24,25,30,33,31,26,34,35,28,22,27,32,29,23],  # G
    [56,55,54,53,52,51,50,49,48,47,46,45,44,43,42,41,40,39,38,37,36],  # H
]

def write_mo(fout, mol, mo_coeff, mo_occ, nkpts, kk, kk_point, kk_weight):
    nmo = mo_coeff.shape[1]
    mo_cart = []
    centers = []
    types = []
    exps = []
    p0 = 0
    for ib in range(mol.nbas):
        ia = mol.bas_atom(ib)
        l = mol.bas_angular(ib)
        es = mol.bas_exp(ib)
        c = mol._libcint_ctr_coeff(ib)
        np, nc = c.shape
        nd = nc*(2*l+1)
        mosub = mo_coeff[p0:p0+nd].reshape(-1,nc,nmo)
        c2s = gto.cart2sph(l)
        mosub = numpy.einsum('yki,cy,pk->pci', mosub, c2s, c)
        mo_cart.append(mosub.transpose(1,0,2).reshape(-1,nmo))

        for t in TYPE_MAP[l]:
            types.append([t]*np)
        ncart = gto.len_cart(l)
        exps.extend([es]*ncart)
        centers.extend([ia+1]*(np*ncart))
        p0 += nd
    mo_cart = numpy.vstack(mo_cart)
    centers = numpy.hstack(centers)
    types = numpy.hstack(types)
    exps = numpy.hstack(exps)
    nprim, nmo = mo_cart.shape

    a = mol.a
    t = mol.get_lattice_Ls()
    t = t[numpy.argsort(lib.norm(t, axis=1))]
    fout.write('CELL\n')
    fout.write(' %11.8f %11.8f %11.8f\n' % (a[0][0], a[0][1], a[0][2]))
    fout.write(' %11.8f %11.8f %11.8f\n' % (a[1][0], a[1][1], a[1][2]))
    fout.write(' %11.8f %11.8f %11.8f\n' % (a[2][0], a[2][1], a[2][2]))
    fout.write('K-POINTS %3d\n' % nkpts)
    fout.write('T-VECTORS %3d\n' % len(t))
    for i in range(len(t)):
        fout.write(' %11.8f %11.8f %11.8f\n' % (t[i][0], t[i][1], t[i][2]))
    fout.write('From PySCF\n')
    fout.write('GAUSSIAN            %3d MOL ORBITALS    %3d PRIMITIVES      %3d NUCLEI\n'
               % (mo_cart.shape[1], mo_cart.shape[0], mol.natm))
    for ia in range(mol.natm):
        x, y, z = mol.atom_coord(ia)
        fout.write('  %-4s %-4d (CENTRE%3d)  %11.8f %11.8f %11.8f  CHARGE = %.1f\n'
                   % (mol.atom_pure_symbol(ia), ia+1, ia+1, x, y, z,
                      mol.atom_charge(ia)))
    for i0, i1 in lib.prange(0, nprim, 20):
        fout.write('CENTRE ASSIGNMENTS  %s\n' % ''.join('%3d'%x for x in centers[i0:i1]))
    for i0, i1 in lib.prange(0, nprim, 20):
        fout.write('TYPE ASSIGNMENTS    %s\n' % ''.join('%3d'%x for x in types[i0:i1]))
    for i0, i1 in lib.prange(0, nprim, 5):
        fout.write('EXPONENTS  %s\n' % ' '.join('%13.7E'%x for x in exps[i0:i1]))

    fout.write('K-POINT %3d WITH COORD %11.8f %11.8f %11.8f WEIGHT %11.8f\n' 
               % (kk, kk_point[0], kk_point[1], kk_point[2], kk_weight))
    for k in range(nmo):
        mo = mo_cart[:,k].real
        moi = mo_cart[:,k].imag
        fout.write('MO  %-4d                  OCC NO = %12.8f\n' %
                   (k+1, mo_occ[k]))
        for i0, i1 in lib.prange(0, nprim, 5):
            fout.write(' %s\n' % ' '.join('%15.8E'%x for x in mo[i0:i1]))
        fout.write('FOLLOW COMPLEX PART\n')
        for i0, i1 in lib.prange(0, nprim, 5):
            fout.write(' %s\n' % ' '.join('%15.8E'%x for x in moi[i0:i1]))
    fout.write('END DATA\n')

def write_mo_k(fout, mol, mo_coeff, mo_occ, kk, kk_point, kk_weight):
    fout.write("K-POINT %3d WITH COORD %11.8f %11.8f %11.8f WEIGHT %11.8f\n" 
               % (kk, kk_point[0], kk_point[1], kk_point[2], kk_weight))
    nmo = mo_coeff.shape[1]
    mo_cart = []
    centers = []
    types = []
    exps = []
    p0 = 0
    for ib in range(mol.nbas):
        ia = mol.bas_atom(ib)
        l = mol.bas_angular(ib)
        es = mol.bas_exp(ib)
        c = mol._libcint_ctr_coeff(ib)
        np, nc = c.shape
        nd = nc*(2*l+1)
        mosub = mo_coeff[p0:p0+nd].reshape(-1,nc,nmo)
        c2s = gto.cart2sph(l)
        mosub = numpy.einsum('yki,cy,pk->pci', mosub, c2s, c)
        mo_cart.append(mosub.transpose(1,0,2).reshape(-1,nmo))

        for t in TYPE_MAP[l]:
            types.append([t]*np)
        ncart = gto.len_cart(l)
        exps.extend([es]*ncart)
        centers.extend([ia+1]*(np*ncart))
        p0 += nd
    mo_cart = numpy.vstack(mo_cart)
    nprim, nmo = mo_cart.shape

    for k in range(nmo):
        mo = mo_cart[:,k].real
        moi = mo_cart[:,k].imag
        fout.write('MO  %-4d                  OCC NO = %12.8f\n' %
                   (k+1, mo_occ[k]))
        for i0, i1 in lib.prange(0, nprim, 5):
            fout.write(' %s\n' % ' '.join('%15.8E'%x for x in mo[i0:i1]))
        fout.write('FOLLOW COMPLEX PART\n')
        for i0, i1 in lib.prange(0, nprim, 5):
            fout.write(' %s\n' % ' '.join('%15.8E'%x for x in moi[i0:i1]))
    fout.write('END DATA\n')

