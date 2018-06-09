#!/usr/bin/env python

import numpy
from pyscf import gto, scf, lib, ao2mo
from pyscf.tools import molden
einsum = lib.einsum

name = 'ch4'

mol = gto.Mole()
mol.basis = 'aug-cc-pvdz'
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
mf = scf.newton(mf)
mf = scf.addons.remove_linear_dep_(mf)
mf.chkfile = name+'.chk'
mf.level_shift = 0.5
mf.kernel()
dm = mf.make_rdm1()
mf.level_shift = 0.0
ehf = mf.kernel(dm)
 
stable_cyc = 3
for i in range(stable_cyc):
    new_mo_coeff = mf.stability(internal=True, external=False)[0]
    if numpy.linalg.norm(numpy.array(new_mo_coeff) - numpy.array(mf.mo_coeff)) < 10**-8:
        lib.logger.info(mf,"* The molecule is stable")
        break
    else:
        lib.logger.info(mf,"* The molecule is unstable")
        n_alpha = numpy.count_nonzero(mf.mo_occ)
        p_alpha = 2.0*numpy.dot(new_mo_coeff[:, :n_alpha], new_mo_coeff.t[:n_alpha])
        mf.kernel(dm0=(p_alpha))
        lib.logger.info(mf,"* Updated SCF energy and orbitals: %16.f" % mf.e_tot)
 
ncore = 0
for ia in range(mol.natm):
    symb = mol.atom_pure_symbol(ia)
    itype = lib.parameters.NUC[symb]
    if (itype == 1 or itype == 2):
      pass
    elif (itype >= 3 and itype <= 10):
      ncore += 1
    elif (itype >= 11 and itype <= 18):
      ncore += 2
    elif (itype >= 19 and itype <= 36):
      ncore += 6
 
nao, nmo = mf.mo_coeff.shape
nocc = mol.nelectron//2 - ncore
nvir = nmo - nocc - ncore
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

den_file = name + '.den'
fspt = open(den_file,'w')
fspt.write('MP2\n')
fspt.write('1-RDM:\n')
occup = 2.0
norb = mol.nelectron//2
for i in range(norb):
    fspt.write('%i %i %.16f\n' % ((i+1), (i+1), occup))
fspt.write('t2_iajb:\n')
for i in range(nocc):
    for j in range(nvir):
        for k in range(nocc):
            for l in range(nvir):
                if (abs(t2[i,j,k,l]) > 1e-8):
                        fspt.write('%i %i %i %i %.10f\n' % ((i+1+ncore), \
                        (j+1+nocc+ncore), (k+1+ncore), (l+1+nocc+ncore), \
                        t2[i,j,k,l]*2.0))
fspt.close()                    

with open(name+'.mol', 'w') as f2:
    molden.header(mol, f2)
    molden.orbital_coeff(mol, f2, mf.mo_coeff, occ=mf.mo_occ)
 
