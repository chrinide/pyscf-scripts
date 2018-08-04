#!/usr/bin/env python

import numpy
from pyscf import gto, scf, lib, ao2mo, dft, tddft
from pyscf.tools import molden
einsum = lib.einsum

mol = gto.Mole()
mol.basis = 'aug-cc-pvqz'
mol.atom = '''
Be  0.0000  0.0000  0.0000
'''
mol.charge = 0
mol.spin = 0
mol.symmetry = 0
mol.verbose = 4
mol.build()

mf = dft.RKS(mol)
mf.xc = 'pbe50'
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

# Grab perturbation tensors in MO basis
origin = ([0.0,0.0,0.0])
mol.set_common_orig(origin)
ao_dip = mol.intor_symmetric('int1e_r', comp=3)
dipoles_xyz = []
for num in range(3):
    Fso = numpy.asarray(ao_dip[num])
    Fia = (co.T).dot(Fso).dot(cv)
    Fia *= -2.0
    dipoles_xyz.append(Fia)

# Build orbital-Hessian
E1  = numpy.einsum('ab,ij->iajb', numpy.diag(ev), numpy.diag(numpy.ones(nocc)))
E1 -= numpy.einsum('ij,ab->iajb', numpy.diag(eo), numpy.diag(numpy.ones(nvir)))
E1 += 4.0*v_iajb
E1 -= v_ijab.swapaxes(1, 2)
E1 -= v_iajb.swapaxes(0, 2)
E1 *= 4.0

def diagonalize(a, b, nroots=5):
    e = numpy.linalg.eig(numpy.bmat([[a        , b       ],
                                     [-b.conj(),-a.conj()]]))[0]
    lowest_e = numpy.sort(e[e > 0])[:nroots]
    return lowest_e

a, b = tddft.TDDFT(mf).get_ab()
a.shape = (nov, nov)
b.shape = (nov, nov)
nroots = 5
print('Direct diagoanlization:', diagonalize(a, b))
print('Reference:', tddft.TDDFT(mf).kernel(nstates=5)[0])

# Since we are time dependent we need to build the full Hessian:
# | A B |      | D  S | |  x |   |  b |
# | B A |  - w | S -D | | -x | = | -b |

# Build A and B blocks
#A11  = numpy.einsum('ab,ij->iajb', numpy.diag(ev), numpy.diag(numpy.ones(nocc)))
#A11 -= numpy.einsum('ij,ab->iajb', numpy.diag(eo), numpy.diag(numpy.ones(nvir)))
#A11 += 2.0*v_iajb
#A11 -= v_ijab.swapaxes(1, 2)
#A11 *= 2.0
A11 = a*2.0

#B11  = -2.0*v_iajb
#B11 += v_iajb.swapaxes(0, 2)
#B11 *= 2.0
B11 = -b*2.0

# The blocks A - B should be equal to E1 / 2
#print('A11 - B11 == E1 / 2?\n',  numpy.allclose(A11 - B11, E1/2.0))

# Reshape and jam it together
#A11.shape = (nov, nov)
#B11.shape = (nov, nov)

Hess1 = numpy.hstack((A11, B11))
Hess2 = numpy.hstack((B11, A11))
Hess = numpy.vstack((Hess1, Hess2))

S11 = numpy.zeros_like(A11)
D11 = numpy.zeros_like(B11)
S11[numpy.diag_indices_from(S11)] = 2.0

S1 = numpy.hstack((S11, D11))
S2 = numpy.hstack((D11, -S11))
S = numpy.vstack((S1, S2))

Hess = Hess.astype(numpy.complex)
S = S.astype(numpy.complex)

dip_x = dipoles_xyz[0].astype(numpy.complex)
B = numpy.hstack((dip_x.ravel(), -dip_x.ravel()))

C6 = numpy.complex(0, 0)
hyper_polar = numpy.complex(0, 0)
leg_points = 10
fdds_lambda = 0.30
print('     Omega      value     weight        sum')

# Integrate over time use a Gauss-Legendre polynomial.
# Shift from [-1, 1] to [0, inf) by the transform  (1 - x) / (1 + x)
for point, weight in zip(*numpy.polynomial.legendre.leggauss(leg_points)):
    if point != 0:
        omega = fdds_lambda *(1.0-point)/(1.0 + point)
        lambda_scale = ((2.0*fdds_lambda)/(point + 1.0)**2)
    else:
        omega = 0.0
        lambda_scale = 0.0

    Hw = Hess - S*complex(0, omega)

    Z =  numpy.linalg.solve(Hw, B)
    value = -numpy.vdot(Z, B)

    if abs(value.imag) > 1.e-13:
        print('Warning value of imaginary part is large', value)

    C6 += (value**2)*weight*lambda_scale
    hyper_polar += value*weight*lambda_scale
    print('% .3e % .3e % .3e % .3e' % (omega, value.real, weight, weight*value.real))

C6 *= 3.0/numpy.pi
print('\nFull C6 Value: %s' % str(C6))

# We can solve static using the above with omega = 0. However a simpler way is
# just to use the reduced form:
dip_x = dip_x.ravel()
static_polar = 2.0*numpy.dot(dip_x, numpy.linalg.solve(A11 - B11, dip_x))

print('Computed values:')
print('Alpha                 % 10.5f' % static_polar.real)
print('C6                    % 10.5f' % C6.real)

print('\nBenchmark values:')
print('C6 He  Limit          % 10.5f' % 1.376)
print('C6 Li+ Limit          % 10.5f' % 0.076)
print('C6 Be  Limit          % 10.5f' % 282.4)

