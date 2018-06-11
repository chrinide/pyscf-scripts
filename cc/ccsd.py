#!/usr/bin/env python

import numpy, sys
from pyscf import gto, scf, cc, lib, ao2mo
from pyscf.tools import molden
sys.path.append('../tools')
import rdm

name = 'n2_ccsd'

mol = gto.Mole()
mol.basis = 'aug-cc-pvdz'
mol.atom = '''
N  0.0000  0.0000  0.5488
N  0.0000  0.0000 -0.5488
'''
mol.verbose = 4
mol.spin = 0
mol.symmetry = 1
mol.charge = 0
mol.build()

mf = scf.RHF(mol)
mf = scf.newton(mf)
mf = scf.addons.remove_linear_dep_(mf)
mf.chkfile = name+'.chk'
#mf.__dict__.update(lib.chkfile.load(name+'.chk', 'scf'))
mf.level_shift = 0.5
mf.conv_tol = 1e-8
mf.kernel()
dm = mf.make_rdm1()
mf.level_shift = 0.0
mf.kernel(dm)

ncore = 2
mcc = cc.CCSD(mf)
mcc.direct = 1
mcc.diis_space = 10
mcc.frozen = ncore
mcc.conv_tol = 1e-6
mcc.conv_tol_normt = 1e-6
mcc.max_cycle = 150
mcc.kernel()

t1norm = numpy.linalg.norm(mcc.t1)
t1norm = t1norm/numpy.sqrt(mol.nelectron-ncore*2)
lib.logger.info(mcc,"* T1 norm should be les than 0.02")
lib.logger.info(mcc,"* T1 norm : %12.6f" % t1norm)
#
#d1mat = numpy.dot(mcc.t1, mcc.t1.T)
#d1o, c = numpy.linalg.eig(d1mat)
#d1 = max(d1o)
#d1norm = numpy.sqrt(d1)
#lib.logger.info(mcc,"* D1 norm should be les than 0.05")
#lib.logger.info(mcc,"* D1 norm : %12.6f" % d1norm)

nao, nmo = mf.mo_coeff.shape
rdm1 = mcc.make_rdm1()
rdm2 = mcc.make_rdm2()
rdm1, rdm2 = rdm.add_inactive_space_to_rdm(mol, nmo, rdm1, rdm2)

eri_mo = ao2mo.kernel(mf._eri, mf.mo_coeff, compact=False)
eri_mo = eri_mo.reshape(nmo,nmo,nmo,nmo)
h1 = reduce(numpy.dot, (mf.mo_coeff.T, mf.get_hcore(), mf.mo_coeff))
ecc =(numpy.einsum('ij,ji->', h1, rdm1)
    + numpy.einsum('ijkl,ijkl->', eri_mo, rdm2)*.5 + mf.mol.energy_nuc())
lib.logger.info(mcc,"* Energy with 1/2-RDM : %.8f" % ecc)    

den_file = name + '.den'
fspt = open(den_file,'w')
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
fspt.close()                    
                 
with open(name+'.mol', 'w') as f2:
    molden.header(mol, f2)
    molden.orbital_coeff(mol, f2, mf.mo_coeff[:,:nmo], occ=mf.mo_occ[:nmo])

