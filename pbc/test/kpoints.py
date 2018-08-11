#!/usr/bin/env python

import numpy
import pyscf.pbc.gto as pbcgto
import pyscf.pbc.scf as pbcscf
import pyscf.pbc.df  as pbcdf
import pyscf.pbc.dft as pbcdft

#from pyscf.pbc.tools import pyscf_ase
#import ase
#import ase.lattice
#from ase.lattice.cubic import Diamond
#ase_atom=Diamond(symbol='C', latticeconstant=3.5668)
#cell = pbcgto.Cell()
#cell.verbose = 4
#cell.atom=pyscf_ase.ase_atoms_to_pyscf(ase_atom)
#cell.a=ase_atom.cell
#cell.basis = 'sto-3g'
##cell.gs = [12,12,12]  # 12 grids on positive x direction, => 25^3 grids in total
#cell.ke_cutoff = 40 # Eh ~ gs = [10,10,10] ~ 21^3 grids in total
#cell.build()

cell = pbcgto.Cell()
cell.atom = '''
He 0 0 1
He 1 0 1
'''
cell.a = numpy.eye(3)*4
cell.gs = [5,5,5]
cell.verbose = 4
cell.basis = 'ccpvdz'
cell.build()
print('The cell volume = %s' % cell.vol)

nks = [2,1,1] 
kpts = cell.make_kpts(nks)
weight = 1./len(kpts)
print('The k-weights = %s' % weight)

df = pbcdf.MDF(cell, kpts)
df.auxbasis = 'weigend'

kmf = pbcscf.KRHF(cell, kpts)
#mf = pbcscf.addons.smearing_(mf, sigma=.1, method='fermi')
kmf.with_df = df
kmf.with_df._cderi_to_save = 'kpoints.h5'
kmf.max_cycle = 150
#mf.with_df._cderi = 'kpoints.h5'
kmf.kernel()
#print('Entropy = %s' % mf.entropy)
#print('Free energy = %s' % mf.e_free)
#print('Zero temperature energy = %s' % ((mf.e_tot+mf.e_free)/2))
dm = kmf.make_rdm1()

coords = numpy.random.random((1,3))

ao_p = pbcdft.numint.eval_ao(cell, coords, kpt=kpts[1], deriv=1)
ao = ao_p[0]
ao_grad = ao_p[1:4]
#print pbcdft.numint.eval_rho(cell, ao_p, dm, xctype='GGA')

ao_p = pbcdft.numint.eval_ao_kpts(cell, coords, kpts=kpts, deriv=1)
ao_kpts = [ao[0] for ao in ao_p]
ao_grad_kpts = [ao[1:4] for ao in ao_p]
#print pbcdft.numint.eval_rho(cell, ao_p, dm, xctype='GGA')

###########################################################
#fspt = open(wfn_file,'w')
#for i in range(len(kpts)):
#    kk_point = kpts[i][:]
#    coeff = kmf.mo_coeff[i][:,kmf.mo_occ[i]>0]
#    occ = kmf.mo_occ[i][kmf.mo_occ[i]>0]
#    if (i==0):
#      wfn_format.write_mo(fspt, cell, coeff, occ, len(kpts), i, kk_point, weight)
#    else :
#      wfn_format.write_mo_k(fspt, cell, coeff, occ, i, kk_point, weight)
#fspt.close()
#for i in range(2):
#    print kmf.mo_occ[i]
#    print kmf.mo_coeff[i]
#    with open(str(i)+'_k.wfn', 'w') as f2:
#        wfn_format.write_mo(f2, cell, kmf.mo_coeff[i])
#from pyscf.pbc.tools import wfn_format
#name = 'gamma'
#wfn_file = name+'.wfn'
#fspt = open(wfn_file,'w')
#coeff = mf.mo_coeff[:,mf.mo_occ>0]
#occ = mf.mo_occ[mf.mo_occ>0]
#wfn_format.write_mo(fspt, cell, coeff, occ)
#a = cell.a
#t = cell.get_lattice_Ls()
#t = t[numpy.argsort(lib.norm(t, axis=1))]
#kpts = numpy.asarray([0.0, 0.0, 0.0])
##expLk = numpy.exp(1j * numpy.asarray(numpy.dot(t, kpts.T), order='C'))
#fspt.write('CRYSTAL\n')
#fspt.write('GAMMA %11.8f %11.8f %11.8f\n' % (kpts[0], kpts[1], kpts[2]))
#fspt.write('CELL\n')
#fspt.write(' %11.8f %11.8f %11.8f\n' % (a[0][0], a[0][1], a[0][2]))
#fspt.write(' %11.8f %11.8f %11.8f\n' % (a[1][0], a[1][1], a[1][2]))
#fspt.write(' %11.8f %11.8f %11.8f\n' % (a[2][0], a[2][1], a[2][2]))
#fspt.write('T-VECTORS %3d\n' % len(t))
#for i in range(len(t)):
#    fspt.write(' %11.8f %11.8f %11.8f\n' % (t[i][0], t[i][1], t[i][2]))
#fspt.close()
#
#s = cell.pbc_intor('cint1e_ovlp_sph')
#k = cell.pbc_intor('cint1e_kin_sph')
#v = cell.pbc_intor('cint1e_nuc_sph')
#enuc = cell.energy_nuc() 
#ekin = einsum('ij,ij->',k,dm)
#pop = einsum('ij,ij->',s,dm)
#elnuce = einsum('ij,ij->',v,dm)
#print('Population : %12.6f' % pop)
#print('Kinetic energy : %12.6f' % ekin)
#print('Nuclear Atraction energy : %12.6f' % elnuce)
#print('Nuclear Repulsion energy : %12.6f' % enuc)
#
#core = mf.get_hcore()
#core = einsum('ij,ij->',core,dm)
#print('1e-energy : %12.6f' % core)
#
#nao = cell.nao_nr()
#eri_ao = ao2mo.restore(1, mf._eri, nao)
#eri_ao = eri_ao.reshape(nao,nao,nao,nao)
#bie1 = einsum('ijkl,ij,kl',eri_ao,dm,dm)*0.5 # J
##bie2 = numpy.einsum('ijkl,il,jk',eri_ao,dm,dm)*0.25 # XC
#pairs1 = einsum('ij,kl,ij,kl->',dm,dm,s,s)*0.5 # J
##pairs2 = numpy.einsum('ij,kl,li,kj->',dm,dm,s,s)*0.25 # XC
##pairs = (pairs1 - pairs2)
#print('Coulomb Pairs : %12.6f' % (pairs1))
##print('XC Pairs : %12.6f' % (pairs2))
##print('Pairs : %12.6f' % pairs)
#print('J energy : %12.6f' % bie1)
##print('XC energy : %12.6f' % -bie2)
##print('EE energy : %12.6f' % (bie1-bie2))
#
##etot = enuc + ekin + elnuce + bie1 - bie2
##print('Total energy : %12.6f' % etot)
#
## En el punto gamma se puede usar toda la artilleria de moleculas
##from pyscf import cc
##mcc = cc.CCSD(mf)
##mcc.direct = True
##mcc.frozen = 0
##mcc.kernel()
#
#########################
#def point(r):
#    ao = pbcdft.numint.eval_ao(cell, r, deriv=1)
#    rhograd = pbcdft.numint.eval_rho(cell, ao, dm, xctype='GGA') 
#    grad = numpy.array([rhograd[1], rhograd[2], rhograd[3]], dtype=numpy.float64)
#    return rhograd[0], grad
#
#print "################# 0.0"
#r = numpy.array([0.00000, 0.00000, 0.00000]) 
#r = numpy.reshape(r, (-1,3))
#rho, grad = point(r)
#print r, rho, grad
#print "################# 0.5"
#r = numpy.array([0.50000, 0.00000, 0.00000]) 
#r = numpy.reshape(r, (-1,3))
#rho, grad = point(r)
#print r, rho, grad
#print "################# 1.0"
#r = numpy.array([1.00000, 0.00000, 0.00000]) 
#r = numpy.reshape(r, (-1,3))
#rho, grad = point(r)
#print r, rho, grad
#print "################# 1.8"
#r = numpy.array([1.80000, 0.00000, 0.00000]) 
#r = numpy.reshape(r, (-1,3))
#rho, grad = point(r)
#print r, rho, grad
#print "################# 3.6"
#r = numpy.array([3.60000, 0.00000, 0.00000]) 
#r = numpy.reshape(r, (-1,3))
#rho, grad = point(r)
###print r, rho, grad
