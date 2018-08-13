#!/usr/bin/env python

import numpy
from pyscf import lib, gto, scf, dft
from pyscf.tools import molden

mol = gto.Mole()
mol.verbose = 4
mol.atom = '''
c   1.217739890298750 -0.703062453466927  0.000000000000000
h   2.172991468538160 -1.254577209307266  0.000000000000000
c   1.217739890298750  0.703062453466927  0.000000000000000
h   2.172991468538160  1.254577209307266  0.000000000000000
c   0.000000000000000  1.406124906933854  0.000000000000000
h   0.000000000000000  2.509154418614532  0.000000000000000
c  -1.217739890298750  0.703062453466927  0.000000000000000
h  -2.172991468538160  1.254577209307266  0.000000000000000
c  -1.217739890298750 -0.703062453466927  0.000000000000000
h  -2.172991468538160 -1.254577209307266  0.000000000000000
c   0.000000000000000 -1.406124906933854  0.000000000000000
h   0.000000000000000 -2.509154418614532  0.000000000000000
'''
mol.basis = 'def2-svpd'
mol.symmetry = 1
mol.build()

a = dft.UKS(mol).density_fit()
a.with_df.auxbasis = 'def2-svp-jkfit'
a.xc = 'pbe0'
a.scf()

# Assign initial guess and reatain (I-MOM)
mo0 = a.mo_coeff
occ = a.mo_occ
occ[0][20]= 0 # this excited state is originated from HOMO(alpha) -> LUMO(alpha)
occ[0][21]= 1 # it is still a singlet state
#occ[1][20]= 0 # this excited state is originated from HOMO(beta) -> LUMO(beta)
#occ[1][21]= 1 # it is still a singlet state
coeffa = mo0[0][:,occ[0]>0]
coeffb = mo0[1][:,occ[1]>0]
mf = dft.UKS(mol).density_fit()
def get_occ(mo_energy=None, mo_coeff=None):

    if mo_energy is None: mo_energy = mf.mo_energy
    if mo_coeff is None: mo_coeff = mf.mo_coeff

    mo_occ = numpy.zeros_like(occ)
    nocc_a = int(numpy.sum(occ[0]))
    nocc_b = int(numpy.sum(occ[1]))
    s_a = reduce(numpy.dot, (coeffa.T, mf.get_ovlp(), mo_coeff[0])) 
    s_b = reduce(numpy.dot, (coeffb.T, mf.get_ovlp(), mo_coeff[1]))
    #choose a subset of mo_coeff, which maximizes <initial|now>
    idx_a = numpy.argsort(numpy.einsum('ij,ij->j', s_a, s_a))
    idx_b = numpy.argsort(numpy.einsum('ij,ij->j', s_b, s_b))
    mo_occ[0][idx_a[-nocc_a:]] = 1.
    mo_occ[1][idx_b[-nocc_b:]] = 1.

    lib.logger.debug(mf,' New alpha occ pattern: %s', mo_occ[0])
    lib.logger.debug(mf,' New beta occ pattern: %s', mo_occ[1])
    lib.logger.debug(mf,' Current alpha mo_energy(sorted) = %s', mo_energy[0])
    lib.logger.debug(mf,' Current beta mo_energy(sorted) = %s', mo_energy[1])

    if (int(numpy.sum(mo_occ[0])) != nocc_a):
        lib.logger.error('mom alpha electron occupation numbers do not match: %d, %d',
                  nocc_a, int(numpy.sum(mo_occ[0])))
    if (int(numpy.sum(mo_occ[1])) != nocc_b):
        lib.logger.error('mom alpha electron occupation numbers do not match: %d, %d',
                  nocc_b, int(numpy.sum(mo_occ[1])))

    return mo_occ
# Assign initial guess and reatain (I-MOM)
mf.xc = 'pbe0'
mf.grids.level = 4
#mf.grids.prune = None
#mf = scf.newton(mf)
mf.with_df.auxbasis = 'def2-svp-jkfit'
dm = mf.make_rdm1(mo0, occ)
mf.get_occ = get_occ
mf.scf(dm)

print('----------------UHF calculation----------------')
print('Excitation energy(UKS): %.3g eV' % ((mf.e_tot - a.e_tot)*27.211))
