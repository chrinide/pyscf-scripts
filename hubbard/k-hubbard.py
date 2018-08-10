#!/usr/bin/env python

import numpy
from pyscf import pbc

t=1.0
U=12 # t/U=0.077 use for FM case

Nx=2
Ny=2
Nele=4
Nkx=15
Nky=15
Nk=Nkx*Nky
Nsite=Nx*Ny

cell = pbc.gto.M(
     unit='B',
     a=[[Nx*1.0,0.0   ,0.0],
        [0.0   ,Ny*1.0,0.0],
        [0.0   ,0.0   ,1.0]],
     verbose=4,
)
cell.nelectron = Nele
kpts = cell.make_kpts([Nkx,Nky,1])

def gen_H_tb(t,Nx,Ny,kvec):
    H = numpy.zeros((Nx,Ny,Nx,Ny),dtype=numpy.complex)
    for i in range(Nx):
        for j in range(Ny):
            if i == Nx-1:
                H[i,j,0   ,j] += numpy.exp(-1j*numpy.dot(numpy.array(kvec),numpy.array([Nx,0])))
            else:
                H[i,j,i+1 ,j] += 1

            if i == 0:
                H[i,j,Nx-1,j] += numpy.exp(-1j*numpy.dot(numpy.array(kvec),numpy.array([-Nx,0])))
            else:
                H[i,j,i-1 ,j] += 1

            if j == Ny-1:
                H[i,j,i,0   ] += numpy.exp(-1j*numpy.dot(numpy.array(kvec),numpy.array([0,Ny])))
            else:
                H[i,j,i,j+1] += 1

            if j == 0:
                H[i,j,i,Ny-1] += numpy.exp(-1j*numpy.dot(numpy.array(kvec),numpy.array([0,-Ny])))
            else:
                H[i,j,i,j-1] += 1
    return -t*H.reshape(Nx*Ny,Nx*Ny)

### get H_tb at a series of kpoints.

def get_H_tb_array(kpts,Nx,Ny,t):
    H_tb_array=[]
    for kpt in kpts:
        H_tb = gen_H_tb(t, Nx, Ny, kpt[:2])
        H_tb_array.append(H_tb)
    return numpy.array(H_tb_array)

# No atoms in the system, exxdiv should be set to None.
kmf = pbc.scf.KUHF(cell, kpts, exxdiv=None)
def get_veff(cell, dm, *args):
    weight = 1./Nk
    j_a = numpy.diag(weight * numpy.einsum('kii->i', dm[0]) * U)
    k_a = numpy.diag(weight * numpy.einsum('kii->i', dm[0]) * U)
    j_b = numpy.diag(weight * numpy.einsum('kii->i', dm[1]) * U)
    k_b = numpy.diag(weight * numpy.einsum('kii->i', dm[1]) * U)
    j = j_a + j_b
    veff_a = numpy.array([j-k_a]*Nk)
    veff_b = numpy.array([j-k_b]*Nk)
    return (veff_a,veff_b)

# Hcore: a (Nk,Nsite,Nsite) array
H_tb_array = get_H_tb_array(kpts,Nx,Ny,t)
kmf.get_hcore = lambda *args: H_tb_array
kmf.get_ovlp = lambda *args: numpy.array([numpy.eye(Nsite)]*Nk)
kmf.get_veff = get_veff

kmf = pbc.scf.addons.smearing_(kmf, sigma=0.2, method='gaussian')
dm_a = numpy.array([numpy.eye(Nsite)]*Nk)
dm_b = dm_a * 0
dm = [dm_a, dm_b]
kmf.kernel(dm)

