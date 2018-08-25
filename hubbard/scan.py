#!/usr/bin/env python

import numpy, sys
from pyscf import gto, scf, mcscf, ao2mo
from pyscf.tools import wfn_format

def hop(tparam, nsites, pbc):
    result = numpy.zeros((nsites,nsites))
    for i in xrange(nsites - 1):
        result[i, i + 1] = tparam
        result[i + 1, i] = tparam
    if pbc == True:
        result[nsites - 1, 0] = tparam
        result[0, nsites - 1] = tparam
    return result

def onsite(uparam, nsites):
    result = numpy.zeros((nsites,nsites,nsites,nsites))
    for i in xrange(nsites):
        result[i, i, i, i] = uparam
    return result

def overlap(nsites):
    return numpy.identity(nsites)

nsites = 4
nelectrons = 4
spin = 0
pbc = True  
results = 'h4.txt'
fs = open(results, 'w')
#fs.write('# U/T EFCI S^2 2S+1 S^2_A 2S+1_A ... Q_A ... F_41 F_42 F_43 F_44\n')
fs.write('# U/T F_41 F_42 F_43 F_44\n')

def run(uparam,tparam):
    s = overlap(nsites)
    h1 = hop(tparam,nsites,pbc) 
    eri = onsite(uparam,nsites)

    mol = gto.Mole()
    mol.nelectron = nelectrons
    mol.verbose = 0
    mol.spin = spin
    mol.symmetry = 0
    mol.charge = 0
    mol.incore_anyway = True
    mol.build()

    mf = scf.RHF(mol)
    mf = scf.addons.frac_occ(mf)
    mf.verbose = 0
    mf.conv_tol = 1e-8
    mf.max_cycle = 150
    mf.diis = True
    mf.diis_space = 12
    mf.get_hcore = lambda *args: h1
    mf.get_ovlp = lambda *args: s
    mf._eri = ao2mo.restore(8, eri, nsites)
    mf.kernel()

    mc = mcscf.CASCI(mf, nsites, nelectrons)
    mc.verbose = 4
    efci = mc.kernel()[0]
    rdm1, rdm2 = mc.fcisolver.make_rdm12(mc.ci, mc.ncas, mc.nelecas) 
    rdm1, rdm2 = mcscf.addons._make_rdm12_on_mo(rdm1, rdm2, mc.ncore, mc.ncas, nsites)

    fs.write('%.7f %.7f ' % (-1.0*uparam/tparam,efci))
    wfn_file = 'h4_%.4f.wfn' % (-1.0*uparam/tparam)
    aom_file = 'h4_%.4f.wfn.aom' % (-1.0*uparam/tparam) 
    edf_file = 'h4_%.4f.edf' % (-1.0*uparam/tparam) 
    den2_file = 'h4_%.4f_2.den' % (-1.0*uparam/tparam) 
    den4_file = 'h4_%.4f_4.den' % (-1.0*uparam/tparam) 
    atms = []
    atms.append(('H',0.0, 1.0, 0.0))
    atms.append(('H',1.0, 1.0, 0.0))
    atms.append(('H',1.0, 0.0, 0.0))
    atms.append(('H',0.0, 0.0, 0.0))
    mol = gto.Mole()
    mol.nelectron = nelectrons
    mol.verbose = 0
    mol.spin = spin
    mol.symmetry = 0
    mol.charge = 0
    mol.basis = 'sto-1g'
    mol.atom = atms
    mol.build()
    nmo = mc.ncore + mc.ncas
    natocc, natorb = numpy.linalg.eigh(-rdm1)
    for i, k in enumerate(numpy.argmax(abs(natorb), axis=0)):
        if natorb[k,i] < 0:
            natorb[:,i] *= -1
    natorb = numpy.dot(mc.mo_coeff[:,:nmo], natorb)
    natocc = -natocc
    with open(wfn_file, 'w') as f2:
        wfn_format.write_mo(f2, mol, natorb, mo_occ=natocc)
        wfn_format.write_coeff(f2, mol, mc.mo_coeff[:,:nmo])
        wfn_format.write_ci_hubbard(f2, mc.ci, mc.ncas, mc.nelecas, ncore=mc.ncore)
    with open(edf_file, 'w') as f2:
        f2.write('0\n')
        f2.write('%s\n' % (aom_file))
        f2.write('%s\n' % (wfn_file))
        f2.write('norecur\n')
        f2.write('probcut 1d-3\n')
        f2.write('ngroup 4\n')
        f2.write('1 1\n')
        f2.write('1 2\n')
        f2.write('1 3\n')
        f2.write('1 4\n')
        f2.write('end\n')
    with open(den2_file, 'w') as f2:
        f2.write('0\n')
        f2.write('%s\n' % (aom_file))
        f2.write('%s\n' % (wfn_file))
        f2.write('ngroup 2\n')
        f2.write('1\n')
        f2.write('2\n')
        f2.write('end\n')
    with open(den4_file, 'w') as f2:
        f2.write('0\n')
        f2.write('%s\n' % (aom_file))
        f2.write('%s\n' % (wfn_file))
        f2.write('ngroup 4\n')
        f2.write('1\n')
        f2.write('2\n')
        f2.write('3\n')
        f2.write('4\n')
        f2.write('end\n')
    with open(aom_file, 'w') as f2:
        for k in range(nsites): # Over atoms == over primitives
            f2.write("%5d <=== AOM within this center\n" % (k+1))
            ij = 0
            for i in range(nsites):
                for j in range(i+1):
                    f2.write(' %16.10f' % (mc.mo_coeff[k,i]*mc.mo_coeff[k,j]))
                    ij += 1
                    if (ij%6 == 0):
                        f2.write("\n")
            f2.write("\n")

    rdm2 = rdm2 - numpy.einsum('ij,kl->ijkl',rdm1,rdm1) 
    rdm1 = reduce(numpy.dot, (mc.mo_coeff, rdm1, mc.mo_coeff.T))
    rdm2 = numpy.dot(mc.mo_coeff, rdm2.reshape(nsites,-1))
    rdm2 = numpy.dot(rdm2.reshape(-1,nsites), mc.mo_coeff.T)
    rdm2 = rdm2.reshape(nsites,nsites,nsites,nsites).transpose(2,3,0,1)
    rdm2 = numpy.dot(mc.mo_coeff, rdm2.reshape(nsites,-1))
    rdm2 = numpy.dot(rdm2.reshape(-1,nsites), mc.mo_coeff.T)
    rdm2 = rdm2.reshape(nsites,nsites,nsites,nsites)
    rdm2 = -rdm2

    pairs2 = numpy.einsum('ijkl,ij,kl->ik',rdm2,s,s, optimize=True)*0.5 # XC
    ia = 3
    for ib in range(ia+1):
        if (ia == ib): 
            factor = 1.0
        if (ia != ib): 
            factor = 2.0
        fs.write('%.7f ' % (2*factor*pairs2[ia,ib]))

    fs.write('\n')

tparam = -0.5
for b in numpy.arange(-15, 18, 0.1):
    uparam =  b
    run(uparam,tparam)

fs.close()
