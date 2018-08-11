#!/usr/bin/env python

import numpy
from pyscf import gto, scf, lib, ao2mo, dft, tddft
from pyscf.tools import molden
from pyscf.data import nist
einsum = lib.einsum

mol = gto.Mole()
mol.basis = '631g'
mol.atom = 'H 0 0 0; F 0 0 1.1'
mol.charge = 0
mol.spin = 0
mol.symmetry = 1
mol.verbose = 4
mol.build()

mf = scf.RHF(mol)
ehf = mf.kernel()

ncore = 0
nao, nmo = mf.mo_coeff.shape
nocc = mol.nelectron//2 - ncore
nvir = nmo - nocc - ncore
nov = nocc * nvir
mo_core = mf.mo_coeff[:,:ncore]
mo_occ = mf.mo_coeff[:,ncore:ncore+nocc]
mo_vir = mf.mo_coeff[:,ncore+nocc:]
co = mo_occ
cv = mo_vir
eo = mf.mo_energy[ncore:ncore+nocc]
ev = mf.mo_energy[ncore+nocc:]

nroots = nocc*nvir

v_ijab = ao2mo.general(mf._eri, (co,co,cv,cv), compact=False)
v_ijab = v_ijab.reshape(nocc,nocc,nvir,nvir)
v_iajb = ao2mo.general(mf._eri, (co,cv,co,cv), compact=False)
v_iajb = v_iajb.reshape(nocc,nvir,nocc,nvir)

def diagonalize(a, b, nroots=nroots):
    e, z = numpy.linalg.eig(numpy.bmat([[ a       , b       ],
                                        [-b.conj(),-a.conj()]]))   
    print e                                        
    lowest_e = numpy.sort(e[e > 0])[:nroots]
    return lowest_e, z

a  = numpy.einsum('ab,ij->iajb', numpy.diag(ev), numpy.diag(numpy.ones(nocc)))
a -= numpy.einsum('ij,ab->iajb', numpy.diag(eo), numpy.diag(numpy.ones(nvir)))
a += 2.0*v_iajb
a -= v_ijab.swapaxes(1, 2)
b  = 2.0*v_iajb
b -= v_iajb.swapaxes(0, 2)
a.shape = (nov, nov)
b.shape = (nov, nov)

e = numpy.linalg.eig(a)[0]
lowest_e = numpy.sort(e[e > 0])[:nroots]
lowest_e = numpy.asarray(lowest_e)*nist.HARTREE2EV
print('CIS (eV):', lowest_e)

e,z = diagonalize(a, b) 
z = numpy.array(z)
e = numpy.asarray(e)*nist.HARTREE2EV
print('TD-HF(eV):', e)

ab1 = a+b
ab2 = a-b
ab = numpy.dot(ab1,ab2)
e = numpy.linalg.eig(ab)[0]
e = numpy.sqrt(e)
lowest_e = numpy.sort(e[e > 0])[:nroots]
lowest_e = numpy.asarray(lowest_e)*nist.HARTREE2EV
print('Alternative TD-HF:', lowest_e)

