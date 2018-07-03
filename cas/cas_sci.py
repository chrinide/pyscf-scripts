#!/usr/bin/env python

import numpy, sys
sys.path.append('../tools')
from pyscf import gto, scf, mcscf, symm, fci, lib
from pyscf.fci import select_ci
from pyscf.mcscf import dmet_cas
import wfn_format

name = 'n2'

mol = gto.Mole()
mol.basis = 'aug-cc-pvdz'
mol.atom = '''
N  0.0000  0.0000  0.5488
N  0.0000  0.0000 -0.5488
    '''
mol.verbose = 4
mol.spin = 0
mol.charge = 0
mol.symmetry = 1
mol.build()

mf = scf.RHF(mol)
mf.chkfile = name+'.chk'
#mf.__dict__.update(lib.chkfile.load(name+'.chk', 'scf'))
mf.level_shift = 0.5
mf.conv_tol = 1e-8
mf = scf.newton(mf)
mf = scf.addons.remove_linear_dep_(mf)
mf.kernel()
dm = mf.make_rdm1()
mf.level_shift = 0.0
ehf = mf.kernel(dm)

aolst1 = ['N 2s']
aolst2 = ['N 2p']
aolst3 = ['N 3s']
aolst4 = ['N 3p']
aolst = aolst1 + aolst2
dm = mf.make_rdm1()
ncas, nelecas, mo = dmet_cas.dmet_cas(mf, dm, aolst, threshold=0.1)

mc = mcscf.CASSCF(mf, ncas, nelecas)
mc.max_cycle_macro = 250
mc.max_cycle_micro = 7
mc.chkfile = name+'.chk'
mc.fcisolver = fci.selected_ci_spin0_symm.SCI(mol)
mc.fix_spin_(shift=.5, ss=0.0000)
mc.fcisolver.ci_coeff_cutoff = 0.0005
mc.fcisolver.select_cutoff = 0.0005
#mc.__dict__.update(scf.chkfile.load(name+'.chk', 'mcscf'))
#mo = lib.chkfile.load(name+'.chk', 'mcscf/mo_coeff')
mc.kernel(mo)

nmo = mc.ncore + mc.ncas
rdm1, rdm2 = mc.fcisolver.make_rdm12(mc.ci, mc.ncas, mc.nelecas) 
rdm1, rdm2 = mcscf.addons._make_rdm12_on_mo(rdm1, rdm2, mc.ncore, mc.ncas, nmo)

orbsym = symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb, mc.mo_coeff[:,:nmo])
natocc, natorb = symm.eigh(-rdm1, orbsym)
for i, k in enumerate(numpy.argmax(abs(natorb), axis=0)):
    if natorb[k,i] < 0:
        natorb[:,i] *= -1
natorb = numpy.dot(mc.mo_coeff[:,:nmo], natorb)
natocc = -natocc

wfn_file = name + '.wfn'
with open(wfn_file, 'w') as f2:
    wfn_format.write_mo(f2, mol, natorb, mo_occ=natocc)
    wfn_format.write_coeff(f2, mol, mc.mo_coeff[:,:nmo])
    wfn_format.write_ci(f2, select_ci.to_fci(mc.ci,mc.ncas,mc.nelecas), mc.ncas, mc.nelecas, ncore=mc.ncore)

