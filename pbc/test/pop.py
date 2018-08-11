#!/usr/bin/env python

import numpy
import h5py
import pyscf.lib.parameters as param
from pyscf.scf import hf
from pyscf.pbc import gto, scf, dft, df
from pyscf.pbc import lib as libpbc
from pyscf.pbc.tools import pbc
from pyscf import lib

name = 'k-lif'

cell = libpbc.chkfile.load_cell(name+'.chk')
cell.ecp = None
mo_coeff_kpts = lib.chkfile.load(name+'.chk', 'scf/mo_coeff')
mo_occ_kpts = lib.chkfile.load(name+'.chk', 'scf/mo_occ')
kpts = lib.chkfile.load(name+'.chk', 'scf/kpts')
nkpts = len(mo_occ_kpts)
dm_kpts = [hf.make_rdm1(mo_coeff_kpts[k], mo_occ_kpts[k]) for k in range(nkpts)]
dm_kpts = lib.asarray(dm_kpts)
nao = cell.nao_nr()
c = numpy.eye(cell.nao_nr())
s_kpts = cell.pbc_intor('cint1e_ovlp_sph', kpts=kpts)
##############################################################################
##############################################################################
label = cell.spheric_labels(False)
print('Monocentric terms')
print('#################')
for k in range(nkpts):
  print "Analysis in kpoint : ", k
  pop = numpy.einsum('ij,ji->i', dm_kpts[k].real,s_kpts[k].real)
  chg1 = numpy.zeros(cell.natm)
  qq1 = numpy.zeros(cell.natm)
  for i, s1 in enumerate(label):
      chg1[s1[0]] += pop[i]
  for ia in range(cell.natm):
      symb = cell.atom_symbol(ia)
      qq1[ia] = cell.atom_charge(ia)-chg1[ia]
      print('Pop, Q, of %d %s = %12.6f %12.6f' % (ia, symb, chg1[ia], qq1[ia]))
print('\nBicentric terms')
print('###############')
for k1 in range(nkpts):
  for k2 in range(k1+1):
    print "Analysis in kpoint : ", k1, k2
    pairs1 = numpy.einsum('ij,kl,ji,lk->ik',dm_kpts[k1].real,dm_kpts[k2].real,s_kpts[k1].real,s_kpts[k2].real)*0.5 # J
    pairs2 = numpy.einsum('ij,kl,il,jk->ik',dm_kpts[k1].real,dm_kpts[k2].real,s_kpts[k1].real,s_kpts[k2].real)*0.25 # XC
    pop = (pairs1 - pairs2)
    chg = numpy.zeros((cell.natm,cell.natm))
    chg1 = numpy.zeros((cell.natm,cell.natm))
    chg2 = numpy.zeros((cell.natm,cell.natm))
    for i, s1 in enumerate(label):
        for j, s2 in enumerate(label):
            factor = 1.0
            chg[s1[0],s2[0]] += pop[i,j]*factor
            chg1[s1[0],s2[0]] += pairs1[i,j]*factor
            chg2[s1[0],s2[0]] += pairs2[i,j]*factor
    check = 0        
    checkii = 0
    checkij = 0
    checkj = 0
    checkxc = 0
    for ia in range(cell.natm):
        symb1 = cell.atom_symbol(ia)
        for ib in range(ia+1):
            symb2 = cell.atom_symbol(ib)
            if (ia == ib): 
                factor = 1.0
                checkii = checkii + chg[ia,ib]
                check = check + chg[ia,ib]
                checkj = checkj + chg1[ia,ib]
                checkxc = checkxc + chg2[ia,ib]
            if (ia != ib): 
                factor = 2.0
                checkj = checkj + factor*chg1[ia,ib]
                checkxc = checkxc + factor*chg2[ia,ib]
                checkij = checkij + factor*chg[ia,ib]
                check = check + factor*chg[ia,ib]
            print('Lambda-Delta, Pairs of  %d %d %s %s = %12.6f %12.6f ' % (ia, ib, symb1, symb2, 2*factor*chg2[ia,ib], chg[ia,ib]))

