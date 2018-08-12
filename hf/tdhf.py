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

v_ijab = ao2mo.general(mf._eri, (co,co,cv,cv), compact=False)
v_ijab = v_ijab.reshape(nocc,nocc,nvir,nvir)
v_iajb = ao2mo.general(mf._eri, (co,cv,co,cv), compact=False)
v_iajb = v_iajb.reshape(nocc,nvir,nocc,nvir)

def diagonalize(a, b, nroots=3):
    e, c = numpy.linalg.eig(\
                 numpy.bmat([[ a       , b       ],
                             [-b.conj(),-a.conj()]]))   
    c = numpy.array(c)
    idx = numpy.where(e>0)
    idx = numpy.asarray(idx[0])
    e = e[idx]
    c = c[:,idx]
    idx = numpy.argsort(e)
    e = e[idx]
    e = e[:nroots]
    c = c[:,idx]
    c = c[:,:nroots]
    return e, c

a  = numpy.einsum('ab,ij->iajb', numpy.diag(ev), \
     numpy.diag(numpy.ones(nocc)))
a -= numpy.einsum('ij,ab->iajb', numpy.diag(eo), \
     numpy.diag(numpy.ones(nvir)))
a += 2.0*v_iajb
a -= v_ijab.swapaxes(1, 2)
b  = 2.0*v_iajb
b -= v_iajb.swapaxes(0, 2)

nroots = 3

a.shape = (nov, nov)
b.shape = (nov, nov)
e, c = diagonalize(a, b, nroots=nroots) 
e = numpy.asarray(e)*nist.HARTREE2EV
lib.logger.info(mf,'TD-HF(eV): %s' % e)

# X^2-Y^2 = 1 TDHF
def norm_xy(z):
    x, y = z.reshape(2,nocc,nvir)
    norm = lib.norm(x)**2 - lib.norm(y)**2
    norm = numpy.sqrt(1.0/norm)
    return x*norm, y*norm

state = 0
lib.logger.info(mf,"Analyzing state : %s" % state)
xy = norm_xy(c[:,state])
coeff = mf.mo_coeff
occ = mf.mo_occ
orbo = coeff[:,occ==2]
orbv = coeff[:,occ==0]
x,y = xy
norm = x**2 - y**2
norm = numpy.sum(norm)
lib.logger.info(mf,"Norm X^2 - Y^2 = 1, value %s" % norm)

ab1 = a+b
ab2 = a-b
ab = numpy.dot(ab1,ab2)
e = numpy.linalg.eig(ab)[0]
e = numpy.sqrt(e)
e = numpy.sort(e[e > 0])[:nroots]
e = numpy.asarray(e)*nist.HARTREE2EV
lib.logger.info(mf,'Alternative TD-HF: %s' % e.real)

