#!/usr/bin/env python

import numpy as np
import pyscf.pbc.gto as pbcgto
import pyscf.pbc.scf as pbcscf
from pyscf.pbc.tools import pyscf_ase
from pyscf.pbc.tools import wfn_format
from mpi4pyscf.pbc import df

from ase.lattice.cubic import Diamond

structure = Diamond(symbol='C', latticeconstant=3.5668)

cell = pbcgto.Cell()
cell.verbose = 4
cell.atom = pyscf_ase.ase_atoms_to_pyscf(structure)
cell.h = structure.cell
cell.basis = 'gth-szv'
cell.pseudo = 'gth-pade'
cell.gs = np.array([2,2,2])
cell.build()

nk = [2,2,2] 
kpts = cell.make_kpts(nk)

kmf = pbcscf.KRHF(cell, kpts)
kmf.kernel()
    
t = cell.get_lattice_Ls()
nocc = np.count_nonzero(kmf.mo_occ != 0)
fermi = np.sort(kmf.mo_energy.ravel())[nocc - 1]
weight = 1./len(kpts)

print(" The cell volume is : %12.8f" % cell.vol)
print(" The Fermi energy is : %12.8f" % fermi)
print(" The k-weight is : %12.8f" % weight)

name = 'diaomond'
wfn_file = name+'.wfn'
fspt = open(wfn_file,'w')
for i in range(len(kpts)):
    kk_point = kpts[i][:]
    coeff = kmf.mo_coeff[i][:,kmf.mo_occ[i]>0]
    occ = kmf.mo_occ[i][kmf.mo_occ[i]>0]
    if (i==0):
      wfn_format.write_mo(fspt, cell, coeff, occ, len(kpts), i, kk_point, weight)
    else :
      wfn_format.write_mo_k(fspt, cell, coeff, occ, i, kk_point, weight)
fspt.close()
