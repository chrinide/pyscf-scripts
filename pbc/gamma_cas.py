#!/usr/bin/env python

import numpy, avas
from pyscf.pbc import gto, scf
from pyscf import fci, ao2mo, mcscf

cell = gto.Cell()
cell.atom='''
H 0.000000000000   0.000000000000   0.000000000000
H 0.000000000000   0.000000000000   1.000000000000
H 0.000000000000   0.000000000000   2.000000000000
H 0.000000000000   0.000000000000   3.000000000000
H 0.000000000000   0.000000000000   4.000000000000
H 0.000000000000   0.000000000000   5.000000000000
H 0.000000000000   0.000000000000   6.000000000000
H 0.000000000000   0.000000000000   7.000000000000
H 0.000000000000   0.000000000000   8.000000000000
H 0.000000000000   0.000000000000   9.000000000000
'''
cell.basis = 'def2-svpd'
cell.precision = 1e-8
cell.dimension = 1
cell.a = '''
10.000000000, 0.000000000, 0.000000000
 0.000000000, 1.000000000, 0.000000000
 0.000000000, 0.000000000, 1.000000000'''
cell.unit = 'A'
cell.verbose = 4
cell.build()

mf = scf.RHF(cell).density_fit(auxbasis='def2-svp-jkfit')
mf.exxdiv = None
ehf = mf.kernel()

ao_labels = ['H 1s']
norb, ne_act, orbs = avas.avas(mf, ao_labels, ncore=0, threshold_occ=0.1, threshold_vir=0.1)

mc = mcscf.CASSCF(mf, norb, ne_act)
mc.fix_spin_(shift=.4, ss=0)
mc.fcisolver = fci.direct_spin0.FCI()
#mc.fcisolver = fci.selected_ci_spin0.SCI()
#mc.fcisolver.ci_coeff_cutoff = 0.005
#mc.fcisolver.select_cutoff = 0.005
mc.kernel(orbs)

nmo = mc.ncore + mc.ncas
rdm1, rdm2 = mc.fcisolver.make_rdm12(mc.ci, mc.ncas, mc.nelecas) 
rdm1, rdm2 = mcscf.addons._make_rdm12_on_mo(rdm1, rdm2, mc.ncore, mc.ncas, nmo)

#natocc, natorb = numpy.linalg.eigh(-rdm1)
#for i, k in enumerate(numpy.argmax(abs(natorb), axis=0)):
#    if natorb[k,i] < 0:
#        natorb[:,i] *= -1
#natorb = numpy.dot(mc.mo_coeff[:,:nmo], natorb)
#natocc = -natocc
#
#wfn_file = name + '.wfn'
#with open(wfn_file, 'w') as f2:
#    wfn_format.write_mo(f2, mol, natorb, mo_occ=natocc)
#    wfn_format.write_coeff(f2, mol, mc.mo_coeff[:,:nmo])
#    wfn_format.write_ci(f2, select_ci.to_fci(mc.ci,mc.ncas,mc.nelecas), mc.ncas, mc.nelecas, ncore=mc.ncore)

