#!/usr/bin/env python

import numpy
from pyscf import gto, scf, mcscf, symm, ao2mo, lib, df
einsum = lib.einsum

mol = gto.Mole()
mol.basis = 'cc-pvtz'
mol.atom = '''
  N  0.0000  0.0000  0.5488
  N  0.0000  0.0000 -0.5488
           '''
mol.verbose = 4
mol.spin = 0
mol.symmetry = 1
mol.symmetry_subgroup = 'D2h'
mol.charge = 0
mol.build()

int3c = df.incore.cholesky_eri(mol, auxbasis='ccpvtz-fit')
mf = scf.density_fit(scf.RHF(mol))
mf.with_df._cderi = int3c
mf.auxbasis = 'cc-pvtz-fit'
mf.conv_tol = 1e-12
mf.direct_scf = 1
mf.level_shift = 0.1
mf.kernel()

mc = mcscf.DFCASSCF(mf, 10, 10)
mc.fcisolver.tol = 1e-8
mc.fcisolver.max_cycle = 250
mc.max_cycle_macro = 250
mc.max_cycle_micro = 7
mc.fcisolver.nroots = 1
mc.kernel()

nao, nmo = mf.mo_coeff.shape
rdm1, rdm2 = mc.fcisolver.make_rdm12(mc.ci, mc.ncas, mc.nelecas) 
rdm1, rdm2 = mcscf.addons._make_rdm12_on_mo(rdm1, rdm2, mc.ncore, mc.ncas, nmo)

naux = mf._cderi.shape[0]
dferi = numpy.empty((naux,nao,nao))
for i in range(naux):
    dferi[i] = lib.unpack_tril(mf._cderi[i])
eri_mo = einsum('rj,Qrs->Qjs', mc.mo_coeff, dferi)
eri_mo = einsum('sb,Qjs->Qjb', mc.mo_coeff, eri_mo)
eri_mo = einsum('Qij,Qkl->ijkl', eri_mo, eri_mo)

h1 = reduce(numpy.dot, (mc.mo_coeff.T, mf.get_hcore(), mc.mo_coeff))
ecc = (numpy.einsum('ij,ij->', h1, rdm1)
    + numpy.einsum('ijkl,ijkl->', eri_mo, rdm2)*.5 + mf.mol.energy_nuc())
lib.logger.info(mc,"* Energy with 1/2-RDM : %.8f" % ecc)    

