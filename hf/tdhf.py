#!/usr/bin/env python

import numpy
from pyscf import gto, scf, lib, ao2mo, dft, tddft
from pyscf.tools import molden
einsum = lib.einsum

mol = gto.Mole()
mol.basis = 'sto-3g'
mol.atom = '''
O
H 1 1.1
H 1 1.1 2 104
'''
mol.charge = 0
mol.spin = 0
mol.symmetry = 0
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

v_ijab = ao2mo.general(mf._eri, (co,co,cv,cv), compact=False)
v_ijab = v_ijab.reshape(nocc,nocc,nvir,nvir)
v_iajb = ao2mo.general(mf._eri, (co,cv,co,cv), compact=False)
v_iajb = v_iajb.reshape(nocc,nvir,nocc,nvir)

def diagonalize(a, b, nroots=5):
    e = numpy.linalg.eig(numpy.bmat([[a        , b       ],
                                     [-b.conj(),-a.conj()]]))[0]
    lowest_e = numpy.sort(e[e > 0])[:nroots]
    return lowest_e

nroots = 5
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
print('CIS:', lowest_e)
print('TD-HF:', diagonalize(a, b))

ab1 = a+b
ab2 = a-b
ab = numpy.dot(ab1,ab2)
e = numpy.linalg.eig(ab)[0]
e = numpy.sqrt(e)
lowest_e = numpy.sort(e[e > 0])[:nroots]
print('Alternative TD-HF:', lowest_e)

excitations = []
for i in range(nocc):
    for a in range(nocc, nmo):
        excitations.append((i, a))

# Initialize CIS matrix.
# The dimensions are the number of possible single excitations
HCIS = numpy.zeros((nov, nov))

# Form matrix elements of shifted CIS Hamiltonian
for p, left_excitation in enumerate(excitations):
    i, a = left_excitation
    for q, right_excitation in enumerate(excitations):
        j, b = right_excitation                                                   
        aa = nmo%a
        bb = nmo%b
        aa = nvir - aa
        bb = nvir - bb
        eri = 2.0*v_iajb[i,aa,j,bb] - v_ijab[i,j,aa,bb]
        HCIS[p, q] = (mf.mo_energy[a] - mf.mo_energy[i]) * (i == j) * (a == b) + eri
ECIS, CCIS = numpy.linalg.eigh(HCIS)            
# Percentage contributions of coefficients for each state vector
percent_contrib = numpy.round(CCIS**2 * 100)

# Print detailed information on significant excitations
print('Info about CIS:')
for state in range(len(ECIS)):
    # Print state, energy
    print('State %3d Energy (Eh) % 10.7f' % (state+1, ECIS[state]))
    for idx, excitation in enumerate(excitations):
        if percent_contrib[idx, state] > 10:
            i, a = excitation
            # Print percentage contribution and the excitation
            print('%4d%% %2d -> %2d' % (percent_contrib[idx, state], i, a))
