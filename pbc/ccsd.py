#!/usr/bin/env python

import numpy
from pyscf.pbc import gto, scf, df, cc
from pyscf import lib
from pyscf.tools import wfn_format

name = 'df-ccsd'

cell = gto.Cell()
cell.unit = 'A'
cell.a = [[4,0,0],[0,1,0],[0,0,1]]
cell.atom = '''
H       0.0000000   0.0000000   0.0000000
H       1.0000000   0.0000000   0.0000000
H       2.0000000   0.0000000   0.0000000
H       3.0000000   0.0000000   0.0000000
''' 
cell.basis = 'sto-6g'
cell.dimension = 1
cell.verbose = 4
cell.build()
 
gdf = df.GDF(cell)

mf = scf.RHF(cell)
#mf.exxdiv = None
mf.with_df = gdf
mf.with_df.auxbasis = 'cc-pvdz-jkfit'
mf.max_cycle = 150
mf.chkfile = name+'.chk'
ehf = mf.kernel()

ncore = 0
#mcc = cc.ccsd.CCSD(mf)
mcc = cc.RCCSD(mf)
mcc.diis_space = 10
mcc.frozen = ncore
mcc.conv_tol = 1e-6
mcc.conv_tol_normt = 1e-6
mcc.max_cycle = 150
mcc.kernel()
rdm1 = mcc.make_rdm1()
rdm2 = mcc.make_rdm2()

nmo = mf.mo_coeff.shape[1]
wfn_file = name + '.wfn'
fspt = open(wfn_file,'w')
wfn_format.write_mo(fspt, cell, mf.mo_coeff, mo_occ=mf.mo_occ, mo_energy=mf.mo_energy)
fspt.write('CCIQA\n')
fspt.write('1-RDM:\n')
for i in range(nmo):
    for j in range(nmo):
        fspt.write('%i %i %.10f\n' % ((i+1), (j+1), rdm1[i,j]))
fspt.write('2-RDM:\n')
for i in range(nmo):
    for j in range(nmo):
        for k in range(nmo):
            for l in range(nmo):
                if (abs(rdm2[i,j,k,l]) > 1e-8):
                        fspt.write('%i %i %i %i %.10f\n' % ((i+1), \
                        (j+1), (k+1), (l+1), rdm2[i,j,k,l]))
a = cell.a
t = cell.get_lattice_Ls()
t = t[numpy.argsort(lib.norm(t, axis=1))]
kpts = numpy.asarray([0.0, 0.0, 0.0])
fspt.write('CRYSTAL\n')
fspt.write('GAMMA %11.8f %11.8f %11.8f\n' % (kpts[0], kpts[1], kpts[2]))
fspt.write('CELL\n')
fspt.write(' %11.8f %11.8f %11.8f\n' % (a[0][0], a[0][1], a[0][2]))
fspt.write(' %11.8f %11.8f %11.8f\n' % (a[1][0], a[1][1], a[1][2]))
fspt.write(' %11.8f %11.8f %11.8f\n' % (a[2][0], a[2][1], a[2][2]))
fspt.write('T-VECTORS %3d\n' % len(t))
for i in range(len(t)):
    fspt.write(' %11.8f %11.8f %11.8f\n' % (t[i][0], t[i][1], t[i][2]))
fspt.close()

