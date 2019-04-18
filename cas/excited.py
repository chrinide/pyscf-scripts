#!/usr/bin/env python

import numpy, os, sys, avas
from pyscf import gto, scf, mcscf, lib, symm, fci
from pyscf.shciscf import shci
from pyscf.tools import wfn_format

name = 'c2h4_01'

mol = gto.Mole()
mol.atom = '''
  C  0.00000000  0.00000000 -1.27395791
  C  0.00000000 -0.00000000  1.27395791
  H  0.00000000  1.75952186  2.34896183
  H  0.00000000 -1.75952186  2.34896183
  H  0.00000000 -1.75952186 -2.34896183
  H  0.00000000  1.75952186 -2.34896183
'''
mol.basis = '3-21g'
mol.unit = 'Bohr'
mol.verbose = 4
mol.spin = 0
mol.symmetry = 'c1'
mol.charge = 0
mol.cart = True
mol.build()

mf = scf.RHF(mol)
mf.max_cycle = 150
mf.chkfile = name+'.chk'
mf.kernel()
mo = mf.mo_coeff

nroots = 4
wghts = numpy.ones(nroots)/nroots
aolst1  = ['C 2px', 'C 2pz']
aolst = aolst1
ncas, nelecas, mo = avas.avas(mf, aolst, threshold_occ=0.1, threshold_vir=1e-5, minao='ano', ncore=2)
 
mch = mcscf.CASSCF(mf, ncas, nelecas)
mch.state_average_(wghts)
mch.fix_spin(ss=0,shift=0.5)
mch.fcisolver.wfnsym = 'A'
mch.fcisolver.spin = 0
mch.kernel(mo)
mo = mch.mo_coeff

ncas = mf.mo_coeff.shape[1] - (mol.nelectron-nelecas)//2
mch = mcscf.CASCI(mf, ncas, nelecas)
mch.fcisolver = fci.direct_spin0_symm.FCI(mol)
mch.fix_spin(ss=0,shift=0.8)
mch.fcisolver.max_cycle = 250
mch.fcisolver.conv_tol = 1e-8
mch.fcisolver.lindep = 1e-14
mch.fcisolver.max_space = 22
mch.fcisolver.level_shift = 0.1
mch.fcisolver.pspace_size = 550
mch.fcisolver.nroots = nroots
mch.fcisolver.spin = 0
mch.fcisolver.wfnsym = 'A'
mch.kernel(mo)

nmo = mch.ncore + mch.ncas
orbsym = symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb, mch.mo_coeff[:,:nmo])
for nr in range(nroots):
    rdm1, rdm2 = mch.fcisolver.make_rdm12(mch.ci[nr], mch.ncas, mch.nelecas)
    rdm1, rdm2 = mcscf.addons._make_rdm12_on_mo(rdm1, rdm2, mch.ncore, mch.ncas, nmo)
    natocc, natorb = symm.eigh(-rdm1, orbsym)
    #natocc, natorb = numpy.linalg.eigh(-rdm1)
    for i, k in enumerate(numpy.argmax(abs(natorb), axis=0)):
        if natorb[k,i] < 0:
            natorb[:,i] *= -1
    natorb = numpy.dot(mch.mo_coeff[:,:nmo], natorb)
    natocc = -natocc
    wfn_file = '_s%i.wfn' % nr 
    wfn_file = name + wfn_file
    with open(wfn_file, 'w') as f2:
        wfn_format.write_mo(f2, mol, natorb, mo_occ=natocc)
        wfn_format.write_coeff(f2, mol, mch.mo_coeff[:,:nmo])
        wfn_format.write_ci(f2, mch.ci[nr], mch.ncas, mch.nelecas, ncore=mch.ncore)

