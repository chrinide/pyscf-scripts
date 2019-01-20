#!/usr/bin/env python

import numpy
from pyscf import gto, scf, dft

mol = gto.Mole()
mol.verbose = 4
mol.atom = [
 ["C",  ( 0.000000,  0.418626, 0.000000)],
 ["H",  (-0.460595,  1.426053, 0.000000)],
 ["O",  ( 1.196516,  0.242075, 0.000000)],
 ["N",  (-0.936579, -0.568753, 0.000000)],
 ["H",  (-0.634414, -1.530889, 0.000000)],
 ["H",  (-1.921071, -0.362247, 0.000000)]
]
mol.basis = {"H": '6-31g',
             "O": '6-31g',
             "N": '6-31g',
             "C": '6-31g',
             }
mol.build()

a = dft.UKS(mol)
a.xc = 'b3lypg'
a.scf()

mo0 = a.mo_coeff
occ0 = a.mo_occ
occ = a.mo_occ

# Assign initial occupation pattern
occ[0][11]=0      # this excited state is originated from HOMO(alpha) -> LUMO(alpha)
occ[0][12]=1      # it is still a singlet state

b = dft.UKS(mol)
#b = scf.addons.frac_occ(b)
b.xc = 'b3lypg'
# Construct new dnesity matrix with new occpuation pattern
dm_u = b.make_rdm1(mo0, occ)
# Apply mom occupation principle
b = scf.addons.mom_occ(b, mo0, occ)
# Start new SCF with new density matrix
b.scf(dm_u)
print b.mo_energy

mo1 = b.mo_coeff
occ1 = b.mo_occ
print('----------------UKS calculation----------------')
print('Excitation energy(UKS): %.3g eV' % ((b.e_tot - a.e_tot)*27.211))
print('Alpha electron occpation pattern of excited state(UKS) : %s' %(b.mo_occ[0]))
print(' Beta electron occpation pattern of excited state(UKS) : %s' %(b.mo_occ[1]))
                                                          
mf = dft.UKS(mol)
mf.xc = 'b3lypg'

# Calculate overlap between two determiant <I|F>
s, x = mf.det_ovlp(mo0, mo1, occ0, occ1)

# Construct density matrix 
dm_s0 = mf.make_rdm1(mo0, occ0)
dm_s1 = mf.make_rdm1(mo1, occ1)
dm_01 = mf.make_asym_dm(mo0, mo1, occ0, occ1, x)

# One-electron part contrbution
h1e = mf.get_hcore(mol)
e1_s0 = numpy.einsum('ji,ji', h1e.conj(), dm_s0[0]+dm_s0[1])
e1_s1 = numpy.einsum('ji,ji', h1e.conj(), dm_s1[0]+dm_s1[1])
e1_01 = numpy.einsum('ji,ji', h1e.conj(), dm_01[0]+dm_01[1])

# Two-electron part contrbution. D_{IF} is asymmetric
vhf_s0 = mf.get_veff(mol, dm_s0)
vhf_s1 = mf.get_veff(mol, dm_s1)
vhf_01 = mf.get_veff(mol, dm_01, hermi=0)

# New total energy: <I|H|I>, <F|H|F>, <I|H|F>
e_s0 = mf.energy_elec(dm_s0, h1e, vhf_s0)
e_s1 = mf.energy_elec(dm_s1, h1e, vhf_s1)
e_01 = mf.energy_elec(dm_01, h1e, vhf_01)

print('The overlap between these two determiants is: %12.8f' % s)
print('E_1e(I),  E_JK(I),  E_tot(I):  %15.7f, %13.7f, %15.7f' % (e1_s0, e_s0[1], e_s0[0]))
print('E_1e(F),  E_JK(F),  E_tot(I):  %15.7f, %13.7f, %15.7f' % (e1_s1, e_s1[1], e_s1[0]))
print('E_1e(IF), E_JK(IF), E_tot(IF): %15.7f, %13.7f, %15.7f' % (e1_01, e_01[1], e_01[0]))
print(' S*<I|H|F> coupling is: %12.7f a.u.' % (e_01[0]*s))
print(' <I|H|F> coupling is: %12.7f a.u.' % (e_01[0]))
print('(0.5*s*H_II+H_FF) is: %12.7f a.u.' % (0.5*s*(e_s0[0]+e_s1[0])))

# Calculate the effective electronic coupling
# V_{IF} = \frac{1}{1-S_{IF}^2}\left| H_{IF} - S_{IF}\frac{H_{II}+H_{FF}}{2} \right|
v01 = s*(e_01[0]-(e_s0[0]+e_s1[0])*0.5)/(1.0 - s*s)
print('The effective coupling is: %7.5f eV' % (numpy.abs(v01)*27.211385) )

