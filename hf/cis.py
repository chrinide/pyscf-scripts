#!/usr/bin/env python

import numpy, sys
sys.path.append('../tools')
from pyscf import gto, scf, lib, ao2mo, dft, tddft
from pyscf.tools import molden
from pyscf.data import nist
import davidson
einsum = lib.einsum

nroots = 3

mol = gto.Mole()
mol.basis = 'sto-3g'
mol.atom = '''
O
H 1 1.1
H 1 1.1 2 104
'''
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
mo_core = mf.mo_coeff[:,:ncore]
mo_occ = mf.mo_coeff[:,ncore:ncore+nocc]
mo_vir = mf.mo_coeff[:,ncore+nocc:]
co = mo_occ
cv = mo_vir
eo = mf.mo_energy[ncore:ncore+nocc]
ev = mf.mo_energy[ncore+nocc:]

v_ijab = ao2mo.general(mf._eri, (co,co,cv,cv), compact=False)
v_ijab = v_ijab.reshape(nocc,nocc,nvir,nvir)
v_iajb = ao2mo.general(mf._eri, (co,cv,co,cv), compact=False)
v_iajb = v_iajb.reshape(nocc,nvir,nocc,nvir)

a  = numpy.einsum('ab,ij->iajb', numpy.diag(ev), \
     numpy.diag(numpy.ones(nocc)))
a -= numpy.einsum('ij,ab->iajb', numpy.diag(eo), \
     numpy.diag(numpy.ones(nvir)))
a += 2.0*v_iajb
a -= v_ijab.swapaxes(1, 2)

nov = nocc * nvir
a.shape = (nov, nov)

e = numpy.linalg.eigh(a)[0]
idx = numpy.argsort(e)
e = e[idx]
e = e[:nroots]
lib.logger.info(mf,'CIS (H): %s' % e)
e = numpy.asarray(e)*nist.HARTREE2EV
lib.logger.info(mf,'CIS (eV): %s' % e)

excitations = []
for i in range(nocc):
    for a in range(nocc, nmo):
        excitations.append((i, a))

# CIS Hamiltonian
shift = 0.0
h = numpy.zeros((nov, nov))
for p, left_excitation in enumerate(excitations):
    i, a = left_excitation
    aa = a - nocc
    for q, right_excitation in enumerate(excitations):
        j, b = right_excitation 
        bb = b - nocc
        eri = 2.0*v_iajb[i,aa,j,bb] - v_ijab[i,j,aa,bb]
        h[p, q] = eri + shift + \
        (mf.mo_energy[a] - mf.mo_energy[i]) * (i==j)*(a==b) 

e, c = numpy.linalg.eigh(h)
idx = numpy.argsort(e)
e = e[idx]
e = e[:nroots]
e = numpy.asarray(e)*nist.HARTREE2EV
c = c[:,idx]
c = c[:,:nroots]

# Percentage contributions of coefficients for each state vector
# X^2 = 1, normalization
percent_contrib = numpy.round(c**2 * 100)
lib.logger.info(mf,'Info about CIS:')
for state in range(nroots):
    lib.logger.info(mf,'State %3d Energy (ev) % 10.7f' % (state+1, e[state]))
    for idx, excitation in enumerate(excitations):
        i, a = excitation
        lib.logger.info(mf,'%s  %2d -> %2d' % (percent_contrib[idx, state], i, a))

