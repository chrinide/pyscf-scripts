#!/usr/bin/env python

import numpy
from functools import reduce
from pyscf import gto, scf, dft, lib
from pyscf.tools import wfn_format

def get_pop(dm, s, lmultipliers):

    label = mol.spheric_labels(False)
    nao = dm.shape[0]
    fock = numpy.zeros_like(dm)
    for ia in range(mol.natm):
        g = numpy.zeros_like(dm)
        iden = numpy.zeros_like(dm)
        for i, s1 in enumerate(label):
            if (s1[0]==ia):
                iden[i,:] = 1.0
        iden = 0.5*(iden+iden.T)
        #print iden
        #tmp = numpy.einsum('ij,ji->ij',iden,s)
        #tmp = numpy.einsum('ij,ji->',tmp,dm)
        #lib.logger.info(mf, 'Pop of %d = %12.6f' % (ia+1, tmp))
        g = numpy.einsum('ij,ji->ij', iden,s)
        fock += lmultipliers[ia]*g

    return fock

def get_cfock(h1e, s1e, vhf, dm, cycle=0, mf_diis=None):
    fock = old_get_fock(h1e, s1e, vhf, dm, cycle, mf_diis)
    fock -= get_pop(dm, s1e, multiplicadores) 
    return fock

name = 'h2'    

mol = gto.Mole()
mol.verbose = 4
mol.atom = '''
H  0.000000000000000  0.000000000000000  0.000000000000000
H  0.000000000000000  0.000000000000000  0.750000000000000
'''
mol.basis = 'sto-3g'
mol.spin = 0
mol.charge = 0
mol.symmetry = 0
mol.build()

mf = scf.RHF(mol)
mf.verbose = 4
mf.kernel()
mf.analyze(with_meta_lowdin=False)

dm = mf.make_rdm1()
multiplicadores = numpy.zeros(mol.natm)
multiplicadores = [0.0,0.2]

my_diis_obj = scf.ADIIS()
my_diis_obj.diis_space = 14
mf = scf.RHF(mol)
mf.conv_tol = 1e-6
mf.max_cycle = 120
mf.diis = my_diis_obj
old_get_fock = mf.get_fock
mf.get_fock = get_cfock 
#mf.verbose = 3
mf.kernel(dm0=dm)
mf.analyze(with_meta_lowdin=False)
dm = mf.make_rdm1()

wfn_file = name + '.wfn'
with open(wfn_file, 'w') as f2:
    wfn_format.write_mo(f2, mol, mf.mo_coeff[:,mf.mo_occ>0], \
    mo_occ=mf.mo_occ[mf.mo_occ>0], mo_energy=mf.mo_energy[mf.mo_occ>0])

