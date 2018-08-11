#!/usr/bin/env python

import numpy, sys
from pyscf import gto, scf, mcscf, ao2mo
from pyscf import fci, hci, lo
from pyscf.fci import select_ci
from pyscf.tools import wfn_format
from pyscf.pbc import scf as pbchf
from pyscf.pbc import gto as pbcgto
from pyscf.pbc import df  as pbcdf
from pyscf.pbc import scf as pbchf

din = 2.0
mol = pbcgto.Cell() 
mol.basis = 'sto-3g'
mol.verbose = 4
mol.atom = '''
    H 0.0 0.0 0.0
    H 0.0 0.0 1.0
    '''
mol.spin = 0
mol.symmetry = 0
mol.charge = 0
mol.dimension = 1
mol.incore_anyway = True
mol.unit='B' 
mol.a=[[din,0.0,0.0],
       [0.0,1.0,0.0],
       [0.0,0.0,1.0]] 
mol.build()
label = mol.spheric_labels(False)
nao = mol.nao_nr()

nks = [8,1,1]
kpts = mol.make_kpts(nks)
kpts -= kpts[0]
mf = pbchf.KRHF(mol, kpts).density_fit()
mf = pbchf.addons.smearing_(mf, sigma=0.2, method='gaussian')
#mf.exxdiv = None
mf.conv_tol = 1e-8
mf.max_cycle = 150
mf.kernel()

dm = mf.make_rdm1()
s = mol.pbc_intor('cint1e_ovlp_sph', kpts=kpts)

nkpts = len(mf.mo_coeff)
for k in range(nkpts):
  chg1 = numpy.zeros(mol.natm)
  qq1 = numpy.zeros(mol.natm)
  print "Analysis in kpoint : ", k, kpts[k]
  popr = numpy.einsum('ij,ji->i', dm[k].real,s[k].real)
  popi = numpy.einsum('ij,ji->i', dm[k].imag,s[k].imag)
  for i, s1 in enumerate(label):
      chg1[s1[0]] += popr[i]+popi[i]
  for ia in range(mol.natm):
      symb = mol.atom_symbol(ia)
      qq1[ia] = mol.atom_charge(ia)-chg1[ia]
      print('Pop, Q, of %d %s = %12.6f %12.6f' % (ia, symb, chg1[ia], qq1[ia]))
