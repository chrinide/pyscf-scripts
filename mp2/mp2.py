#!/usr/bin/env python

import numpy
from pyscf import gto, scf, lib, ao2mo
from pyscf.tools import molden
einsum = lib.einsum

name = 'ch4'

mol = gto.Mole()
mol.basis = 'aug-cc-pvtz'
mol.atom = '''
C  0.0000  0.0000  0.0000
H  0.6276  0.6276  0.6276
H  0.6276 -0.6276 -0.6276
H -0.6276  0.6276 -0.6276
H -0.6276 -0.6276  0.6276
'''
mol.charge = 0
mol.spin = 0
mol.symmetry = 1
mol.verbose = 4
mol.build()

mf = scf.RHF(mol)
mf.chkfile = name+'.chk'
ehf = mf.kernel()

nao, nmo = mf.mo_coeff.shape
ncore = 1
nocc = mol.nelectron//2 - ncore
nvir = nmo - nocc - ncore

# https://journals.aps.org/pra/pdf/10.1103/PhysRevA.73.012510
#hcore = mf.get_hcore()
#for i in range(nmo):
#    shift = reduce(numpy.dot, (mf.mo_coeff[:,i].T, hcore, mf.mo_coeff[:,i]))
#    mf.mo_energy[i] = 0.5*(mf.mo_energy[i]+shift)
 
mo_core = mf.mo_coeff[:,:ncore]
mo_occ = mf.mo_coeff[:,ncore:ncore+nocc]
mo_vir = mf.mo_coeff[:,ncore+nocc:]
co = mo_occ
cv = mo_vir
eo = mf.mo_energy[ncore:ncore+nocc]
ev = mf.mo_energy[ncore+nocc:]
eri_mo = ao2mo.general(mf._eri, (co,cv,co,cv), compact=False)
eri_mo = eri_mo.reshape(nocc,nvir,nocc,nvir)
lib.logger.info(mf,"* Core orbitals: %d" % ncore)
lib.logger.info(mf,"* Virtual orbitals: %d" % (len(ev)))
 
e_denom = 1.0/(eo.reshape(-1,1,1,1)-ev.reshape(-1,1,1)+eo.reshape(-1,1)-ev)
t2 = numpy.zeros((nocc,nvir,nocc,nvir))
t2 = 2.0*einsum('iajb,iajb->iajb', eri_mo, e_denom)
t2 -= einsum('ibja,iajb->iajb', eri_mo, e_denom)
e_mp2 = numpy.einsum('iajb,iajb->', eri_mo, t2, optimize=True)
lib.logger.info(mf,"!*** E(MP2): %12.8f" % e_mp2)
lib.logger.info(mf,"!**** E(HF+MP2): %12.8f" % (e_mp2+ehf))

#den_file = name + '.den'
#fspt = open(den_file,'w')
#fspt.write('MP2\n')
#fspt.write('1-RDM:\n')
#occup = 2.0
#norb = mol.nelectron//2
#for i in range(norb):
#    fspt.write('%i %i %.16f\n' % ((i+1), (i+1), occup))
#fspt.write('t2_iajb:\n')
#for i in range(nocc):
#    for j in range(nvir):
#        for k in range(nocc):
#            for l in range(nvir):
#                if (abs(t2[i,j,k,l]) > 1e-8):
#                        fspt.write('%i %i %i %i %.10f\n' % ((i+1+ncore), \
#                        (j+1+nocc+ncore), (k+1+ncore), (l+1+nocc+ncore), \
#                        t2[i,j,k,l]*2.0))
#fspt.close()                    
#
#with open(name+'.mol', 'w') as f2:
#    molden.header(mol, f2)
#    molden.orbital_coeff(mol, f2, mf.mo_coeff, occ=mf.mo_occ)
# 
