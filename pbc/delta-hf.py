#!/usr/bin/env python

import numpy
from functools import reduce
from pyscf.pbc import df  as pbcdf
from pyscf.pbc import scf as pbcscf
from pyscf.pbc import gto as pbcgto
from pyscf.pbc import dft as pbcdft
from pyscf import lo, lib, ao2mo
from pyscf.lo import orth
einsum = lib.einsum

name = 'gamma-delta-hf'

cell = pbcgto.Cell()
cell.atom = '''C     0.      0.      0.    
               C     0.8917  0.8917  0.8917
               C     1.7834  1.7834  0.    
               C     2.6751  2.6751  0.8917
               C     1.7834  0.      1.7834
               C     2.6751  0.8917  2.6751
               C     0.      1.7834  1.7834
               C     0.8917  2.6751  2.6751'''
cell.a = numpy.eye(3)*3.5668
cell.basis = 'gth-szv'
cell.pseudo = 'gth-pade'
cell.verbose = 4
cell.symmetry = 0
cell.mesh = [10,10,10]
cell.build()

mf = pbcscf.RHF(cell)
mf.with_df = pbcdf.AFTDF(cell)
mf.with_df._cderi_to_save = name+'.h5'
#mf.with_df._cderi = name+'.h5'
mf.kernel()

nao = cell.nao_nr()
dm = mf.make_rdm1()
mo = mf.mo_coeff
s = cell.pbc_intor('cint1e_ovlp_sph')

#c = lo.nao.nao(cell, mf, restore=False)
#c = numpy.eye(cell.nao_nr())
#mo = numpy.linalg.solve(c, mf.mo_coeff)
#dm = mf.make_rdm1(mo, mf.mo_occ)

pop = einsum('ij,ji->',s,dm)
lib.logger.info(mf, 'Population : %s' % pop)

pairs1 = einsum('ij,kl,ij,kl->',dm,dm,s,s)*0.5 # J
pairs2 = einsum('ij,kl,li,kj->',dm,dm,s,s)*0.25 # XC
pairs = (pairs1 - pairs2)
lib.logger.info(mf, 'Coulomb Pairs : %12.6f' % (pairs1))
lib.logger.info(mf, 'XC Pairs : %12.6f' % (pairs2))
lib.logger.info(mf, 'Pairs : %12.6f' % pairs)

##############################################################################
##############################################################################
label = cell.spheric_labels(False)
lib.logger.info(mf,'\nPopulation and charges in NAO basis')
lib.logger.info(mf,'###################################')
pop = einsum('ij,ji->i', dm,s)
chg1 = numpy.zeros(cell.natm)
qq1 = numpy.zeros(cell.natm)
for i, s1 in enumerate(label):
    chg1[s1[0]] += pop[i]
for ia in range(cell.natm):
    symb = cell.atom_symbol(ia)
    qq1[ia] = cell.atom_charge(ia)-chg1[ia]
    lib.logger.info(mf, 'Pop, Q, of %d %s = %12.6f %12.6f' \
    % (ia+1, symb, chg1[ia], qq1[ia]))

lib.logger.info(mf,'\nSume rules test')
lib.logger.info(mf,'###############')
lib.logger.info(mf,'Sum of charges : %12.6f' % sum(chg1))

lib.logger.info(mf,'\nLocalization/delocalication in NAO basis')
lib.logger.info(mf,'########################################')
pairs1 = einsum('ij,kl,ij,kl->ik',dm,dm,s,s)*0.5 # J
pairs2 = einsum('ij,kl,li,kj->ik',dm,dm,s,s)*0.25 # XC
pop = (pairs1 - pairs2)
chg = numpy.zeros((cell.natm,cell.natm))
chg1 = numpy.zeros((cell.natm,cell.natm))
chg2 = numpy.zeros((cell.natm,cell.natm))
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
for ia in range(cell.natm):
    symb1 = cell.atom_symbol(ia)
    for ib in range(ia+1):
        symb2 = cell.atom_symbol(ib)
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

