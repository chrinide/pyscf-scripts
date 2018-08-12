#!/usr/bin/env python

import numpy
from functools import reduce
from pyscf import gto, scf, lib, df, lo, dft

mol = gto.Mole()
mol.atom = '''
O      0.000000      0.000000      0.118351
H      0.000000      0.761187     -0.469725
H      0.000000     -0.761187     -0.469725
           '''
mol.verbose = 4
mol.spin = 0
mol.symmetry = 1
mol.charge = 0
mol.basis = 'def2-tzvppd'
mol.build()

mf = dft.RKS(mol)
mf.conv_tol = 1e-8
mf.grids.radi_method = dft.mura_knowles
mf.grids.becke_scheme = dft.stratmann
mf.grids.level = 4
mf.grids.prune = None
mf.xc = 'pbe0'
mf.kernel()

c = lo.orth_ao(mol, 'nao', scf_method=mf)
mo = numpy.linalg.solve(c, mf.mo_coeff)
dm = mf.make_rdm1(mo, mf.mo_occ)

s = mol.intor('int1e_ovlp')
s = reduce(numpy.dot, (c.T,s,c))
label = mol.spheric_labels(False)

lib.logger.info(mf,'\nPopulation and charges in NAO basis')
lib.logger.info(mf,'###################################')
pop = numpy.einsum('ij,ji->i', dm,s)
chg1 = numpy.zeros(mol.natm)
qq1 = numpy.zeros(mol.natm)
for i, s1 in enumerate(label):
    chg1[s1[0]] += pop[i]
for ia in range(mol.natm):
    symb = mol.atom_symbol(ia)
    qq1[ia] = mol.atom_charge(ia)-chg1[ia]
    lib.logger.info(mf, 'Pop, Q, of %d %s = %12.6f %12.6f' \
    % (ia+1, symb, chg1[ia], qq1[ia]))

lib.logger.info(mf,'\nSume rules test')
lib.logger.info(mf,'###############')
lib.logger.info(mf,'Sum of charges : %12.6f' % sum(chg1))

lib.logger.info(mf,'\nLocalization/delocalication in NAO basis')
lib.logger.info(mf,'########################################')
pairs1 = numpy.einsum('ij,kl,ij,kl->ik',dm,dm,s,s)*0.5 # J
pairs2 = numpy.einsum('ij,kl,li,kj->ik',dm,dm,s,s)*0.25 # XC
pop = (pairs1 - pairs2)
chg = numpy.zeros((mol.natm,mol.natm))
chg1 = numpy.zeros((mol.natm,mol.natm))
chg2 = numpy.zeros((mol.natm,mol.natm))
for i, s1 in enumerate(label):
    for j, s2 in enumerate(label):
        factor = 1.0
        chg[s1[0],s2[0]] += pop[i,j]*factor
        chg1[s1[0],s2[0]] += pairs1[i,j]*factor
        chg2[s1[0],s2[0]] += pairs2[i,j]*factor
check = 0        
checkii = 0
checkij = 0
checkj = 0
checkxc = 0
for ia in range(mol.natm):
    symb1 = mol.atom_symbol(ia)
    for ib in range(ia+1):
        symb2 = mol.atom_symbol(ib)
        if (ia == ib): 
            factor = 1.0
            checkii = checkii + chg[ia,ib]
            check = check + chg[ia,ib]
            checkj = checkj + chg1[ia,ib]
            checkxc = checkxc + chg2[ia,ib]
        if (ia != ib): 
            factor = 2.0
            checkj = checkj + factor*chg1[ia,ib]
            checkxc = checkxc + factor*chg2[ia,ib]
            checkij = checkij + factor*chg[ia,ib]
            check = check + factor*chg[ia,ib]
        lib.logger.info(mf, \
        'Lambda-Delta, Pairs of  %d %d %s %s = %12.6f %12.6f' \
        % (ia+1, ib+1, symb1, symb2, 2.0*factor*chg2[ia,ib], \
        chg[ia,ib]))

lib.logger.info(mf, '\n##############')
lib.logger.info(mf, 'Sum rules test')
lib.logger.info(mf, '##############')
lib.logger.info(mf, 'Total Coulomb pairs : %12.6f' % checkj)
lib.logger.info(mf, 'Total XC pairs : %12.6f' % checkxc)
lib.logger.info(mf, 'Total intra pairs : %12.6f' % checkii)
lib.logger.info(mf, 'Total inter pairs : %12.6f' % checkij)
lib.logger.info(mf, 'Total pairs : %12.6f' % check)

