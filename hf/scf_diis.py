#!/usr/bin/env python

import numpy
from pyscf import lib, gto, scf, dft

mol = gto.Mole()
mol.atom = '''
  O     0.000000000000     0.000000000000     0.224348285559
  H    -1.423528800232     0.000000000000    -0.897393142237
  H     1.423528800232     0.000000000000    -0.897393142237
'''
mol.basis = '6-31g*'
mol.cart = True
mol.verbose = 4
mol.spin = 0
mol.symmetry = 'c1'
mol.unit = 'B'
mol.charge = 0
mol.build()

mf = scf.RHF(mol)
mf.kernel()

s = mol.intor('int1e_ovlp')
v = mol.intor('int1e_nuc')
t = mol.intor('int1e_kin')
eri_ao = mol.intor('int2e')

def make_rdm1(mo_coeff, mo_occ):
    mocc = mo_coeff[:,mo_occ>0]
    return numpy.dot(mocc*mo_occ[mo_occ>0], mocc.T.conj())
def eigh(h):
    e, c = numpy.linalg.eigh(h)
    idx = numpy.argmax(abs(c.real), axis=0)
    c[:,c[idx,numpy.arange(len(e))].real<0] *= -1
    return e, c
def lowdin(s):
    e, v = eigh(s)
    idx = e > 1e-15
    return numpy.dot(v[:,idx]/numpy.sqrt(e[idx]), v[:,idx].conj().T)

nao = s.shape[0]
h = t + v
nocc = mol.nelectron//2
a = lowdin(s)    
hc = a.dot(h).dot(a)
e, c2 = eigh(hc)
c = a.dot(c2)
occ = numpy.zeros(nao)
occ[:nocc] = 2.0
dm = make_rdm1(c,occ)
enuc = mol.energy_nuc()

e_1el = numpy.einsum('pq,pq->', h, dm) + enuc
print('one-electron energy = %4.16f' % e_1el)

e = 0.0
eold = 0.0
dold = numpy.zeros_like(dm)
maxiter = 50
e_conv = 1.0e-8
d_conv = 1.0e-6
fock_list = []
diis_error = []
diis_size = 10
diis_start = 2

for scf_iter in range(1,maxiter+1):
    j = numpy.einsum('pqrs,rs->pq', eri_ao, dm)
    k = numpy.einsum('prqs,rs->pq', eri_ao, dm)
    veff = j - 0.5*k
    f = h + veff

    # DIIS error
    diis_e = lib.einsum('ij,jk,kl->il', f, dm, s)
    diis_e -= lib.einsum('ij,jk,kl->il', s, dm, f)
    diis_e = a.dot(diis_e).dot(a)
    fock_list.append(f)
    diis_error.append(diis_e)
    drms = numpy.mean(diis_e**2)**0.5

    scf_e = numpy.einsum('pq,pq->', h, dm) + enuc
    scf_e += numpy.einsum('pq,pq->', veff, dm)*0.5
    print('scf iteration %3d: energy = %4.16f   de = % 1.5e   drms = %1.5e' 
          % (scf_iter, scf_e, (scf_e - eold), drms))

    if (abs(scf_e - eold) < e_conv) and (drms < d_conv):
        break

    eold = scf_e
    dold = dm

    if scf_iter >= diis_start:
        # limit size of diis vector
        diis_count = len(fock_list)
        if diis_count > diis_size:
            # remove oldest vector
            del fock_list[0]
            del diis_error[0]
            diis_count -= 1

        # build error matrix b
        b = numpy.empty((diis_count + 1, diis_count + 1))
        b[-1, :] = -1
        b[:, -1] = -1
        b[-1, -1] = 0
        for num1, e1 in enumerate(diis_error):
            for num2, e2 in enumerate(diis_error):
                if num2 > num1: continue
                val = numpy.einsum('ij,ij->', e1, e2)
                b[num1, num2] = val
                b[num2, num1] = val

        # normalize
        b[:-1, :-1] /= numpy.abs(b[:-1, :-1]).max()

        # build residual vector
        resid = numpy.zeros(diis_count + 1)
        resid[-1] = -1

        # solve pulay equations
        ci = numpy.linalg.solve(b, resid)

        # calculate new fock matrix as linear
        # combination of previous fock matrices
        f = numpy.zeros_like(f)
        for num, c in enumerate(ci[:-1]):
            f += c * fock_list[num]

    # Diagonalize Fock matrix
    fp = a.dot(f).dot(a)
    e, c2 = eigh(fp)
    c = a.dot(c2)
    dm = make_rdm1(c,occ)

    if scf_iter == maxiter:
        raise RuntimeError("maximum number of scf cycles exceeded.")

