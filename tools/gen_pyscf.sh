#!/bin/bash

points="01p2 01p3 01p4 01p5 01p6 01p7 01p8 01p9 02p0 02p2 02p4 02p6 02p8
03p0 03p2 03p4 03p6 03p8 04p0 04p2 04p4 05p0 05p5 05p8 05p9 06p0 06p1
06p2 06p5 07p0 08p0 09p0 10p0 12p0"

for i in $points; do
cat << EOF > lif_${i}.py
#!/usr/bin/env python

import numpy
from pyscf import gto, scf, mcscf, lib, symm, fci, dft
from pyscf.tools import wfn_format
from pyscf import solvent

name = 'lif_$i'

mol = gto.Mole()
mol.atom = '''
  Li  0.0000  0.0000  0.0000
  F   0.0000  0.0000  `echo "$i" | sed -r 's/[p]+/./g'`
'''
mol.basis = 'aug-cc-pvdz'
mol.verbose = 4
mol.spin = 0
mol.symmetry = 1
mol.charge = 0
mol.build()

mf = scf.RHF(mol)
mf.chkfile = 'guess.chk'
mf = solvent.ddCOSMO(mf)
mf.with_solvent.lebedev_order = 53
mf.with_solvent.lmax = 10
mf.with_solvent.max_cycle = 150
mf.with_solvent.conv_tol = 1e-7
mf.with_solvent.grids.radi_method = dft.mura_knowles
mf.with_solvent.grids.becke_scheme = dft.stratmann
mf.with_solvent.grids.level = 4
mf.with_solvent.grids.prune = None
mf.max_cycle = 150
mf.__dict__.update(scf.chkfile.load('guess.chk', 'scf'))
dm = mf.make_rdm1()
mf.kernel(dm)

mo = mf.mo_coeff
nroots = 3
wghts = numpy.ones(nroots)/nroots
ncas = 12
nelecas = 8
 
mch = mcscf.CASSCF(mf, ncas, nelecas)
mch = mch.state_average_(wghts)
mch.chkfile = 'guess.chk'
mch.fix_spin(ss=0,shift=0.8)
mch.fcisolver.spin = 0
mch.fcisolver.max_cycle = 250
mch.fcisolver.conv_tol = 1e-8
mch.fcisolver.lindep = 1e-14
mch.fcisolver.max_space = 24
mch.fcisolver.level_shift = 0.03
mch.fcisolver.pspace_size = 850
mch = solvent.ddCOSMO(mch)
mch.with_solvent.lebedev_order = 53
mch.with_solvent.lmax = 10 
mch.with_solvent.max_cycle = 80
mch.with_solvent.conv_tol = 5e-7
mch.with_solvent.grids.radi_method = dft.mura_knowles
mch.with_solvent.grids.becke_scheme = dft.stratmann
mch.with_solvent.grids.level = 4
mch.with_solvent.grids.prune = None
mo = lib.chkfile.load('guess.chk', 'mcscf/mo_coeff')
mo = mcscf.project_init_guess(mch, mo)
mch.kernel(mo)

mo = mch.mo_coeff
nelecas = 8
ncas = 12

mch = mcscf.CASCI(mf, ncas, nelecas)
mch.fix_spin(ss=0,shift=0.8)
mch.fcisolver.max_cycle = 250
mch.fcisolver.conv_tol = 1e-8
mch.fcisolver.lindep = 1e-14
mch.fcisolver.max_space = 24
mch.fcisolver.level_shift = 0.03
mch.fcisolver.pspace_size = 850
mch.fcisolver.nroots = nroots
mch.fcisolver.spin = 0
mch = solvent.ddCOSMO(mch)
mch.with_solvent.lebedev_order = 53
mch.with_solvent.lmax = 10 
mch.with_solvent.max_cycle = 80
mch.with_solvent.conv_tol = 5e-7
mch.with_solvent.grids.radi_method = dft.mura_knowles
mch.with_solvent.grids.becke_scheme = dft.stratmann
mch.with_solvent.grids.level = 4
mch.with_solvent.grids.prune = None
mch.kernel(mo)

nmo = mch.ncore + mch.ncas
orbsym = symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb, mch.mo_coeff[:,:nmo])
for nr in range(nroots):
    rdm1, rdm2 = mch.fcisolver.make_rdm12(mch.ci[nr], mch.ncas, mch.nelecas)
    rdm1, rdm2 = mcscf.addons._make_rdm12_on_mo(rdm1, rdm2, mch.ncore, mch.ncas, nmo)
    natocc, natorb = symm.eigh(-rdm1, orbsym)
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

EOF
done
