#!/usr/bin/env python

import numpy, sys
sys.path.append('../tools')
from pyscf import gto, scf, mcscf, symm, fci, lib
from pyscf.tools import molden
import avas

name = 'n2'

mol = gto.Mole()
mol.basis = 'aug-cc-pvdz'
mol.atom = '''
N  0.0000  0.0000  0.5488
N  0.0000  0.0000 -0.5488
    '''
mol.verbose = 4
mol.spin = 0
mol.charge = 0
mol.symmetry = 1
mol.build()

mf = scf.RHF(mol)
mf = scf.newton(mf)
mf = scf.addons.remove_linear_dep_(mf)
mf.chkfile = name+'.chk'
#mf.__dict__.update(lib.chkfile.load(name+'.chk', 'scf'))
mf.level_shift = 0.5
mf.conv_tol = 1e-8
mf.kernel()
dm = mf.make_rdm1()
mf.level_shift = 0.0
ehf = mf.kernel(dm)

stable_cyc = 3
for i in range(stable_cyc):
    new_mo_coeff = mf.stability(internal=True, external=False)[0]
    if numpy.linalg.norm(numpy.array(new_mo_coeff) - numpy.array(mf.mo_coeff)) < 10**-8:
        lib.logger.info(mf,"* The molecule is stable")
        break
    else:
        lib.logger.info(mf,"* The molecule is unstable")
        n_alpha = numpy.count_nonzero(mf.mo_occ)
        p_alpha = 2.0*numpy.dot(new_mo_coeff[:, :n_alpha], new_mo_coeff.t[:n_alpha])
        mf.kernel(dm0=(p_alpha))
        lib.logger.info(mf,"* Updated SCF energy and orbitals: %16.f" % mf.e_tot)

ncore = 0
for ia in range(mol.natm):
    symb = mol.atom_pure_symbol(ia)
    itype = lib.parameters.NUC[symb]
    if (itype == 1 or itype == 2):
      pass
    elif (itype >= 3 and itype <= 10):
      ncore += 1
    elif (itype >= 11 and itype <= 18):
      ncore += 2
    elif (itype >= 19 and itype <= 36):
      ncore += 6

aolst1 = ['N 2s']
aolst2 = ['N 2p']
aolst3 = ['N 3s']
aolst4 = ['N 3p']
aolst = aolst1 + aolst2 + aolst3 + aolst4
ncas, nelecas, mo = avas.kernel(mf, aolst, threshold_occ=0.1, threshold_vir=0.01, minao='minao', ncore=ncore)

mc = mcscf.CASSCF(mf, ncas, nelecas)
mc.max_cycle_macro = 250
mc.max_cycle_micro = 7
mc.chkfile = name+'.chk'
mc.fcisolver = fci.direct_spin0_symm.FCI()
mc.fix_spin_(shift=.5, ss=0)
#mc.__dict__.update(scf.chkfile.load(name+'.chk', 'mcscf'))
#mo = lib.chkfile.load(name+'.chk', 'mcscf/mo_coeff')
mc.kernel(mo)

nmo = mc.ncore + mc.ncas
rdm1, rdm2 = mc.fcisolver.make_rdm12(mc.ci, mc.ncas, mc.nelecas) 
rdm1, rdm2 = mcscf.addons._make_rdm12_on_mo(rdm1, rdm2, mc.ncore, mc.ncas, nmo)

den_file = name + '.den'
fspt = open(den_file,'w')
fspt.write('CCIQA\n')
fspt.write('1-RDM:\n')
for i in range(nmo):
    for j in range(nmo):
        fspt.write('%i %i %.10f\n' % ((i+1), (j+1), rdm1[i,j]))
fspt.write('2-RDM:\n')
for i in range(nmo):
    for j in range(nmo):
        for k in range(nmo):
            for l in range(nmo):
                if (abs(rdm2[i,j,k,l]) > 1e-8):
                        fspt.write('%i %i %i %i %.10f\n' % ((i+1), \
                        (j+1), (k+1), (l+1), rdm2[i,j,k,l]))
fspt.close()                    
    
with open(name+'.mol', 'w') as f2:
    molden.header(mol, f2)
    molden.orbital_coeff(mol, f2, mc.mo_coeff[:,:nmo], occ=mf.mo_occ[:nmo])

