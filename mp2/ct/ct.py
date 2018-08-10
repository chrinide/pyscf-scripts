#!/usr/bin/python

import numpy
import scipy.linalg

from pyscf import gto, scf, mcscf, ao2mo, fci, mp
from pyscf.lib import logger
from pyscf import lib
einsum = lib.einsum

name = 'ct'

r = 1.1
mol = gto.Mole()
mol.verbose = 4
mol.atom = [
    ['N', ( 0., 0.    , -r/2 )],
    ['N', ( 0., 0.    ,  r/2)],]
mol.basis = 'cc-pcVTZ'
mol.symmetry = 1
mol.build()
    
mf = scf.RHF(mol)
mf.chkfile = name+'.chk'
#mf.__dict__.update(lib.chkfile.load(name+'.chk', 'scf'))
mf.kernel()

pt2 = mp.MP2(mf)
pt2.kernel()

mo = mf.mo_coeff
mc = mcscf.CASSCF(mf, 10, 10)
mc.max_cycle_macro = 250
mc.max_cycle_micro = 7
mc.chkfile = name+'.chk'
mc.fcisolver = fci.direct_spin0_symm.FCI()
mc.fix_spin_(shift=.5, ss=0)
#mo = lib.chkfile.load(name+'.chk', 'mcscf/mo_coeff')
emc = mc.mc1step(mo)[0]
    
mo = mc.mo_coeff
nmo = mo.shape[1]
ncore = mc.ncore
ncas = mc.ncas
nocc = mc.ncore + mc.ncas
nvir = mo.shape[1] - nocc

# transform the CAS natural orbitals
rdm1 = mc.fcisolver.make_rdm1(mc.ci, mc.ncas, mc.nelecas)
occ, ucas = scipy.linalg.eigh(rdm1)
logger.info(mc, 'Natural occs')
logger.info(mc, str(occ))
natocc = numpy.zeros(mo.shape[1])
natocc[:ncore] = 1
natocc[ncore:nocc] = occ[::-1] * .5
natorb = numpy.dot(mo[:,ncore:nocc], ucas[:,::-1])
natorb = numpy.hstack((mo[:,:ncore], natorb, mo[:,nocc:]))
    
# integrals
h1e_mo = mc.get_hcore()
h1e_mo = reduce(numpy.dot, (natorb.T, h1e_mo, natorb))
v2e = ao2mo.full(mf._eri, natorb)
v2e = ao2mo.restore(1, v2e, nmo) 
v2e = v2e.transpose(0,2,1,3) # to physics notation

# transform Hamiltonian to quasiparticle frame
beta = numpy.sqrt(natocc)
alpha = numpy.sqrt(1.0-natocc)
#
tmpa = h1e_mo + \
       einsum('prqr,r->pq', v2e, 2.0*beta**2) - \
       einsum('prrq,r->pq', v2e, beta**2)
h1e_bogo = einsum('p,pq,q->pq', alpha, tmpa, alpha)
tmpb = h1e_mo + \
       einsum('prqr,r->pq', v2e, 2.0*beta**2) - \
       einsum('prrq,r->pq', v2e, beta**2)
h1e_bogo -= einsum('p,pq,q->pq', beta, tmpb, beta)
tmp3 = einsum('pqrr,p,q,r,r->pq', v2e, alpha, beta, alpha, beta) + \
       einsum('qprr,q,p,r,r->pq', v2e, alpha, beta, alpha, beta)
h1e_bogo -= tmp3
wpqrs = einsum('pqsr,p,q,r,s->pqrs', v2e, alpha, alpha, beta, beta)

# make diagonal and transform to semi-canonical basis
e1, c1 = scipy.linalg.eigh(h1e_bogo[:ncore,:ncore])
e2, c2 = scipy.linalg.eigh(h1e_bogo[ncore:nocc,ncore:nocc])
e3, c3 = scipy.linalg.eigh(h1e_bogo[nocc:,nocc:])
e_bogo = numpy.hstack((e1,e2,e3))
c = numpy.zeros((nmo,nmo))
c[:ncore,:ncore] = c1
c[ncore:nocc,ncore:nocc] = c2
c[nocc:,nocc:] = c3
wpqrs = ao2mo.full(wpqrs, c)
    
# Core-to-external
dpq = e_bogo[nocc:][:,None] + e_bogo[nocc:]
drs = e_bogo[:ncore][:,None] + e_bogo[:ncore]
dpqrs = dpq.reshape(-1,1) + drs.reshape(-1)
dpqrs = 1.0/dpqrs.reshape(nvir,nvir,ncore,ncore)
ectmp2_ccee = -2.0*einsum('pqrs,pqrs->', wpqrs[nocc:,nocc:,:ncore,:ncore]**2, dpqrs)
tmp = einsum('pqrs,pqsr->pqrs', wpqrs[nocc:,nocc:,:ncore,:ncore], wpqrs[nocc:,nocc:,:ncore,:ncore])
ectmp2_ccee += einsum('pqrs,pqrs->', tmp, dpqrs)
logger.info(mc, "CT-MP2 Doubles Energy (CCEE): %s" % ectmp2_ccee)
    
# Core-to-active
dpq = e_bogo[ncore:nocc][:,None] + e_bogo[ncore:nocc]
drs = e_bogo[:ncore][:,None] + e_bogo[:ncore]
dpqrs = dpq.reshape(-1,1) + drs.reshape(-1)
dpqrs = 1.0/dpqrs.reshape(ncas,ncas,ncore,ncore)
ectmp2_ccaa = -2.0*einsum('pqrs,pqrs->', wpqrs[ncore:nocc,ncore:nocc,:ncore,:ncore]**2, dpqrs)
tmp = einsum('pqrs,pqsr->pqrs', wpqrs[ncore:nocc,ncore:nocc,:ncore,:ncore], wpqrs[ncore:nocc,ncore:nocc,:ncore,:ncore])
ectmp2_ccaa += einsum('pqrs,pqrs->', tmp, dpqrs)
logger.info(mc, "CT-MP2 Doubles Energy (CCAA): %s" % ectmp2_ccaa)
    
# Active-to-external
dpq = e_bogo[nocc:][:,None] + e_bogo[nocc:]
drs = e_bogo[ncore:nocc][:,None] + e_bogo[ncore:nocc]
dpqrs = dpq.reshape(-1,1) + drs.reshape(-1)
dpqrs = 1.0/dpqrs.reshape(nvir,nvir,ncas,ncas)
ectmp2_aaee = -2.0*einsum('pqrs,pqrs', wpqrs[nocc:,nocc:,ncore:nocc,ncore:nocc]**2, dpqrs)
tmp = einsum('pqrs,pqsr->pqrs', wpqrs[nocc:,nocc:,ncore:nocc,ncore:nocc], wpqrs[nocc:,nocc:,ncore:nocc,ncore:nocc])
ectmp2_aaee += einsum('pqrs,pqrs->', tmp, dpqrs)
logger.info(mc, "CT-MP2 Doubles Energy (AAEE): %s" % ectmp2_aaee)

e_corr = (ectmp2_ccee + ectmp2_ccaa + ectmp2_aaee) 
logger.info(mc, "CT-MP2 Correlation Energy: %s" % e_corr)
e = emc + ectmp2_ccee + ectmp2_ccaa + ectmp2_aaee
logger.info(mc, "CASSCF + CT-MP2 Energy: %s" % e)

