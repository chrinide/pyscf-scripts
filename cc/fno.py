#!/usr/bin/env python

import numpy, sys, os
sys.path.append('../tools')
from pyscf import gto, scf, lib, ao2mo, ao2mo, cc
from pyscf.tools import molden
import utils
einsum = lib.einsum

name = 'f2_ccsd_fno'
mol = gto.Mole()
mol.basis = 'aug-cc-pvtz'
mol.atom = '''
F      0.000000      0.000000      0.018162
F      0.000000      0.000000      1.393738
'''
mol.charge = 0
mol.spin = 0
mol.symmetry = 0
mol.verbose = 4
mol.build()
#
mf = scf.RHF(mol)
mf.chkfile = name+'.chk'
ehf = mf.kernel()
#
ncore = 2
lib.logger.info(mf,"* Core orbitals: %d" % ncore)
nocc = mol.nelectron//2 - ncore
mo_core = mf.mo_coeff[:,:ncore]
mo_occ = mf.mo_coeff[:,ncore:ncore+nocc]
co = mo_occ
eo = mf.mo_energy[ncore:ncore+nocc]
nocc = len(eo)
#
cv,ev = utils.getfno(mf,ncore,thresh_vir=1e-4)
nvir = len(ev)
e_denom = 1.0/(eo.reshape(-1,1,1,1)-ev.reshape(-1,1,1)+eo.reshape(-1,1)-ev)
eri_mo = ao2mo.general(mf._eri, (co,cv,co,cv), compact=False)
eri_mo = eri_mo.reshape(nocc,nvir,nocc,nvir)
lib.logger.info(mf,"* Virtual orbitals: %d" % (len(ev)))
#
rdm2_mp2 = numpy.zeros((nocc,nvir,nocc,nvir))
rdm2_mp2 = 2.0*einsum('iajb,iajb->iajb', eri_mo, e_denom)
rdm2_mp2 -= einsum('ibja,iajb->iajb', eri_mo, e_denom)
e_mp2 = numpy.einsum('iajb,iajb->', eri_mo, rdm2_mp2, optimize=True)
lib.logger.info(mf,"!*** E(MP2): %12.8f" % e_mp2)
lib.logger.info(mf,"!**** E(HF+MP2): %12.8f" % (e_mp2+ehf))
#
mo_vir = cv
coeff = numpy.hstack([mo_core,mo_occ,mo_vir])
nao, nmo = coeff.shape
nocc = mol.nelectron//2
occ = numpy.zeros(nmo)
for i in range(nocc):
    occ[i] = 2.0
#
mycc = cc.CCSD(mf, mo_coeff=coeff, mo_occ=occ)
mycc.direct = True
mycc.frozen = 2
mycc.kernel()
#
rdm1 = mycc.make_rdm1()
rdm2 = mycc.make_rdm2()
eri_mo = ao2mo.kernel(mf._eri, coeff[:,:nmo], compact=False)
eri_mo = eri_mo.reshape(nmo,nmo,nmo,nmo)
h1 = reduce(numpy.dot, (coeff[:,:nmo].T, mf.get_hcore(), coeff[:,:nmo]))
ecc =(numpy.einsum('ij,ji->', h1, rdm1)
    + numpy.einsum('ijkl,ijkl->', eri_mo, rdm2)*.5 + mf.mol.energy_nuc())
lib.logger.info(mycc,"* Energy with 1/2-RDM : %.8f" % ecc)    

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
    molden.orbital_coeff(mol, f2, coeff[:,:nmo], occ=occ[:nmo])

