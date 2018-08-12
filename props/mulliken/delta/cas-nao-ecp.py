#!/usr/bin/env python

import numpy, sys
from pyscf import scf, gto, mcscf, lo, ao2mo, fci, lib
sys.path.append('../../tools')
import avas

mol = gto.Mole()
mol.verbose = 4
mol.atom = '''
C 0.0 0.0 0.0
O 0.0 0.0 1.12
    '''
mol.basis= {'C':'crenbl','O':'crenbl'}
mol.ecp = {'C':'crenbl','O':'crenbl'} 
mol.symmetry = 1
mol.spin = 0
mol.charge = 0
mol.build()

mf = scf.RHF(mol)
ehf = mf.kernel()

aolst1 = ['C 2s']
aolst2 = ['C 2p']
aolst = aolst1 + aolst2
ncas, nelecas, mo = avas.avas(mf, aolst, threshold_occ=0.1, threshold_vir=0.001, minao='ano', verbose=4, ncore=0)

mc = mcscf.CASSCF(mf, ncas, nelecas)
mc.fcisolver.tol = 1e-8
mc.fcisolver.max_cycle = 100
mc.max_cycle_macro = 20
mc.max_cycle_micro = 7
mc.fcisolver.nroots = 1
#mc.fcisolver = fci.select_ci_spin0.SCI(mol)
#mc.fcisolver = fci.direct_spin0_symm.FCI(mol)
mc.fix_spin_(shift=.5, ss=0)
#mc.fcisolver.ci_coeff_cutoff = 0.0001
#mc.fcisolver.select_cutoff = 0.0001
mc.kernel(mo)
mo = mc.mo_coeff

nao = mol.nao_nr()
dm1, dm2 = mc.fcisolver.make_rdm12(mc.ci,mc.ncas,mc.nelecas)
rdm1, rdm2 = mcscf.addons._make_rdm12_on_mo(dm1,dm2,mc.ncore,mc.ncas,nao)

#c = lo.orth_ao(mol, 'nao', scf_method=mc)
c = lo.orth_ao(mol, 'meta_lowdin')
mo = numpy.linalg.solve(c, mc.mo_coeff)

rdm2 = rdm2 - numpy.einsum('ij,kl->ijkl',rdm1,rdm1) 
rdm1 = reduce(numpy.dot, (mo, rdm1, mo.T))
rdm2 = numpy.dot(mo, rdm2.reshape(nao,-1))
rdm2 = numpy.dot(rdm2.reshape(-1,nao), mo.T)
rdm2 = rdm2.reshape(nao,nao,nao,nao).transpose(2,3,0,1)
rdm2 = numpy.dot(mo, rdm2.reshape(nao,-1))
rdm2 = numpy.dot(rdm2.reshape(-1,nao), mo.T)
rdm2 = rdm2.reshape(nao,nao,nao,nao)
rdm2 = -rdm2

s = mol.intor('cint1e_ovlp_sph')
s = reduce(numpy.dot, (c.T,s,c))

label = mol.spheric_labels(False)

lib.logger.info(mf,'\nPopulation and charges in NAO basis')
lib.logger.info(mf,'###################################')
pop = numpy.einsum('ij,ij->i', rdm1,s)
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
pairs1 = numpy.einsum('ij,kl,ij,kl->ik',rdm1,rdm1,s,s)*0.5 # J
pairs2 = numpy.einsum('ijkl,ij,kl->ik',rdm2,s,s)*0.5 # XC
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
checkj = 0        
checkxc = 0        
checkii = 0
checkij = 0
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

