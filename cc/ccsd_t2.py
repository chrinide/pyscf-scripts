#!/usr/bin/env python

import numpy, sys
from functools import reduce
from pyscf import gto, scf, cc, lib, ao2mo
from pyscf.tools import molden

name = 'n2_ccsd_t2'

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
e_corr, t1, t2 = mcc.kernel()

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
nocc = mol.nelectron/2 - ncore
nvir = nmo - nocc - ncore

def tran_rdm2(t1, t2):
    nocc, nvir = t1.shape
    tau = numpy.empty((1,nocc,nvir,nvir))
    rdm2 = numpy.empty((nocc,nocc,nvir,nvir))
    for p0 in range(nocc):
        p1 = p0 + 1
        from pyscf.cc import _ccsd
        _ccsd.make_tau(t2[p0:p1], t1[p0:p1], t1, 1, out=tau)
        theta = tau*2 - tau.transpose(0,1,3,2)
        rdm2[p0,:,:,:] = theta[0,:,:,:]
    rdm2 = numpy.einsum('ijab->iajb', rdm2) # rdm2 as iajb
    return rdm2*2.0

rdm2 = tran_rdm2(t1, t2)

den_file = name + '.den'
fspt = open(den_file,'w')
fspt.write('MP2\n')
fspt.write('La matriz D es:\n')
occup = 2.0
norb = mol.nelectron//2
for i in range(norb):
    fspt.write('%i %i %.16f\n' % ((i+1), (i+1), occup))
fspt.write('La matriz d es:\n')
for i in range(nocc):
    for j in range(nvir):
        for k in range(nocc):
            for l in range(nvir):
                if (abs(rdm2[i,j,k,l]) > 1e-8):
                        fspt.write('%i %i %i %i %.10f\n' % ((i+1+ncore), \
                        (j+1+nocc+ncore), (k+1+ncore), (l+1+nocc+ncore), rdm2[i,j,k,l]))
fspt.close()                    

with open(name+'.mol', 'w') as f2:
    molden.header(mol, f2)
    molden.orbital_coeff(mol, f2, mf.mo_coeff[:,:nmo], occ=mf.mo_occ[:nmo])

