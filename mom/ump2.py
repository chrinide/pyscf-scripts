#!/usr/bin/env python

import numpy
from pyscf import gto, scf, mp, lib, ao2mo
from pyscf.tools import molden
einsum = lib.einsum

name = 'ump2'

mol = gto.Mole()
mol.verbose = 4
mol.atom = '''
O      0.000000      0.000000      0.118351
H      0.000000      0.761187     -0.469725
H      0.000000     -0.761187     -0.469725
'''
mol.basis = 'sto-3g'
mol.symmetry = 0
mol.build()

a = scf.UHF(mol)
a.scf()

####### This can be removed ######
frozen = [[0],[0]] # 1sa and 1sb
pt2 = mp.UMP2(a)
pt2.frozen = frozen
pt2.kernel()
rdm1a, rdm1b = pt2.make_rdm1()
rdm2aa, rdm2ab, rdm2bb = pt2.make_rdm2()

mo_a = a.mo_coeff[0]
mo_b = a.mo_coeff[1]
nmoa = mo_a.shape[1]
nmob = mo_b.shape[1]
eriaa = ao2mo.kernel(a._eri, mo_a, compact=False).reshape([nmoa]*4)
eribb = ao2mo.kernel(a._eri, mo_b, compact=False).reshape([nmob]*4)
eriab = ao2mo.kernel(a._eri, (mo_a,mo_a,mo_b,mo_b), compact=False)
eriab = eriab.reshape([nmoa,nmoa,nmob,nmob])
hcore = a.get_hcore()
h1a = reduce(numpy.dot, (mo_a.T.conj(), hcore, mo_a))
h1b = reduce(numpy.dot, (mo_b.T.conj(), hcore, mo_b))
e1 = einsum('ij,ji', h1a, rdm1a)
e1+= einsum('ij,ji', h1b, rdm1b)
e1+= einsum('ijkl,ijkl', eriaa, rdm2aa)*0.5
e1+= einsum('ijkl,ijkl', eriab, rdm2ab)
e1+= einsum('ijkl,ijkl', eribb, rdm2bb)*0.5
e1+= mol.energy_nuc()
lib.logger.info(pt2,"* Ground state Energy with 1/2-RDM : %.8f" % e1) 
####### This can be removed ######

mo0 = a.mo_coeff
occ = a.mo_occ

# Assign initial occupation pattern
occ[0][4]=0 # this excited state is originated from HOMO(alpha) -> LUMO(alpha)
occ[0][5]=1 # it is still a singlet state

mf = scf.UHF(mol)
dm_u = mf.make_rdm1(mo0, occ)
mf = scf.addons.mom_occ(mf, mo0, occ)
mf.scf(dm_u)

frozen = [[0],[0]] # 1sa and 1sb
pt2 = mp.UMP2(mf)
pt2.frozen = frozen
pt2.kernel()

rdm1a, rdm1b = pt2.make_rdm1()
rdm2aa, rdm2ab, rdm2bb = pt2.make_rdm2()
rdm2ba = rdm2ab.transpose(2,3,0,1)

occ_a = mf.mo_occ[0]
occ_b = mf.mo_occ[1]
mo_a = mf.mo_coeff[0]
mo_b = mf.mo_coeff[1]
nmoa = mo_a.shape[1]
nmob = mo_b.shape[1]

####### This can be removed ######
eriaa = ao2mo.kernel(mf._eri, mo_a, compact=False).reshape([nmoa]*4)
eribb = ao2mo.kernel(mf._eri, mo_b, compact=False).reshape([nmob]*4)
eriab = ao2mo.kernel(mf._eri, (mo_a,mo_a,mo_b,mo_b), compact=False)
eriab = eriab.reshape([nmoa,nmoa,nmob,nmob])
hcore = mf.get_hcore()
h1a = reduce(numpy.dot, (mo_a.T.conj(), hcore, mo_a))
h1b = reduce(numpy.dot, (mo_b.T.conj(), hcore, mo_b))
e1 = einsum('ij,ji', h1a, rdm1a)
e1+= einsum('ij,ji', h1b, rdm1b)
e1+= einsum('ijkl,ijkl', eriaa, rdm2aa)*0.5
e1+= einsum('ijkl,ijkl', eriab, rdm2ab)
e1+= einsum('ijkl,ijkl', eribb, rdm2bb)*0.5
e1+= mol.energy_nuc()
lib.logger.info(pt2,"* Excited state Energy with 1/2-RDM : %.8f" % e1) 
####### This can be removed ######

coeff = numpy.hstack([mo_a,mo_b])
occ = numpy.hstack([occ_a,occ_b])
with open(name+'.mol', 'w') as f2:
    molden.header(mol, f2)
    molden.orbital_coeff(mol, f2, coeff, occ=occ)

# Aqui cuidado para hacer una prueba rapida los bucles
# Se recorren en alpha por ejemplo porque hay el mismo
# numero de alpha y beta congelados, si fuese distinto
# habria que cabiar
nmo = nmoa
den_file = name + '.den'
fspt = open(den_file,'w')
fspt.write('CCIQA\n')
fspt.write('1-RDM:\n')
for i in range(nmo):
    for j in range(nmo):
        fspt.write('%i %i %.10f\n' % ((i+1), (j+1), rdm1a[i,j]))
        fspt.write('%i %i %.10f\n' % ((i+1+nmo), (j+1+nmo), rdm1b[i,j]))
fspt.write('2-RDM:\n')
for i in range(nmo):
    for j in range(nmo):
        for k in range(nmo):
            for l in range(nmo):
                if (abs(rdm2aa[i,j,k,l]) > 1e-8):
                    fspt.write('%i %i %i %i %.10f\n' % ((i+1), \
                    (j+1), (k+1), (l+1), rdm2aa[i,j,k,l]))
                if (abs(rdm2bb[i,j,k,l]) > 1e-8):
                    fspt.write('%i %i %i %i %.10f\n' % ((i+1+nmo), \
                    (j+1+nmo), (k+1+nmo), (l+1+nmo), rdm2bb[i,j,k,l]))
                if (abs(rdm2ab[i,j,k,l]) > 1e-8):
                    fspt.write('%i %i %i %i %.10f\n' % ((i+1), \
                    (j+1), (k+1+nmo), (l+1+nmo), rdm2ab[i,j,k,l]))
                if (abs(rdm2ba[i,j,k,l]) > 1e-8):
                    fspt.write('%i %i %i %i %.10f\n' % ((i+1+nmo), \
                    (j+1+nmo), (k+1), (l+1), rdm2ba[i,j,k,l]))
fspt.close()                    

