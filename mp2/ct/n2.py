#!/usr/bin/python

import numpy
import scipy.linalg

from pyscf import gto, scf, mcscf, ao2mo,mp
from pyscf.lib import logger
from pyscf import lib
einsum = lib.einsum

r = 1.1
mol = gto.Mole()
mol.verbose = 4
mol.atom = [
    ['N', ( 0., 0.    , -r/2 )],
    ['N', ( 0., 0.    ,  r/2)],]
mol.basis = 'cc-pcVTZ'
mol.build()
    
mf = scf.RHF(mol)
ehf = mf.scf()

pt2 = mp.MP2(mf)
pt2.frozen = 1
pt2.kernel()

mc = mcscf.CASSCF(mf, 10, 10)
emc = mc.mc1step()[0]
    
#  natural orbitals
def make_natorb(casscf):
    fcivec = casscf.ci
    mo = casscf.mo_coeff
    ncore = casscf.ncore
    ncas = casscf.ncas
    nelecas = casscf.nelecas
    nocc = ncore + ncas

    casdm1 = casscf.fcisolver.make_rdm1(fcivec, ncas, nelecas)
    # alternatively, return alpha, beta 1-pdm seperately
    #casdm1a,casdm1b = casscf.fcisolver.make_rdm1s(fcivec, ncas, nelecas)

    occ, ucas = scipy.linalg.eigh(casdm1)
    logger.info(casscf, 'Natural occs')
    logger.info(casscf, str(occ))
    natocc = numpy.zeros(mo.shape[1])
    natocc[:ncore] = 1
    natocc[ncore:nocc] = occ[::-1] * .5

    # transform the CAS natural orbitals from MO repr. to AO repr.
    # mo[AO_idx,MO_idx]
    natorb_in_cas = numpy.dot(mo[:,ncore:nocc], ucas[:,::-1])
    natorb_on_ao = numpy.hstack((mo[:,:ncore], natorb_in_cas, mo[:,nocc:]))
    return natorb_on_ao, natocc
    
natorb, natocc = make_natorb(mc)
    
#  integrals
ncore = mc.ncore
ncas = mc.ncas
nocc = ncore + ncas
mo = mc.mo_coeff
nvir = mo.shape[1] - nocc

h1e_ao = mc.get_hcore()
h1e_mo = reduce(numpy.dot, (natorb.T, h1e_ao, natorb))

v2e = ao2mo.incore.full(mf._eri, natorb) # v2e has 4-fold symmetry now
    
nmo = natorb.shape[1]
v2e = ao2mo.restore(1, v2e, nmo) # to remove 4-fold symmetry, turn v2e to n**4 array
v2e = v2e.transpose(0,2,1,3) # to physics notation

beta = numpy.sqrt(natocc)
alpha = numpy.sqrt(1.0-natocc)

tmpa = h1e_mo + einsum('prqr,r->pq', v2e, 2.0 * beta**2) - einsum('prrq,r->pq', v2e, beta**2)
h1e_bogo = einsum('p,pq,q->pq', alpha, tmpa, alpha)

tmpb = h1e_mo + einsum('prqr,r->pq', v2e, 2.0 * beta**2) - einsum('prrq,r->pq', v2e, beta**2)
h1e_bogo -= einsum('p,pq,q->pq', beta, tmpb, beta)

tmp3 = einsum('pqrr,p,q,r,r->pq', v2e, alpha, beta, alpha, beta) \
     + einsum('qprr,q,p,r,r->pq', v2e, alpha, beta, alpha, beta)
h1e_bogo -= tmp3

wpqrs = einsum('pqsr,p,q,r,s->pqrs', v2e, alpha, alpha, beta, beta)

# transform to semi-canonical basis
e1, c1 = scipy.linalg.eigh(h1e_bogo[:ncore,:ncore])
e2, c2 = scipy.linalg.eigh(h1e_bogo[ncore:nocc,ncore:nocc])
e3, c3 = scipy.linalg.eigh(h1e_bogo[nocc:,nocc:])

e_bogo = numpy.hstack((e1,e2,e3))

c = numpy.zeros((nmo,nmo))
c[:ncore,:ncore] = c1
c[ncore:nocc,ncore:nocc] = c2
c[nocc:,nocc:] = c3

wpqrs = einsum('pqrs,px->xqrs', wpqrs, c)
wpqrs = einsum('pqrs,qx->pxrs', wpqrs, c)
wpqrs = einsum('pqrs,rx->pqxs', wpqrs, c)
wpqrs = einsum('pqrs,sx->pqrx', wpqrs, c)
    
# Core-to-external
dpq = e_bogo[nocc:][:,None] + e_bogo[nocc:]
drs = e_bogo[:ncore][:,None] + e_bogo[:ncore]
dpqrs = dpq.reshape(-1,1) + drs.reshape(-1)
dpqrs = 1.0/dpqrs.reshape(nvir,nvir,ncore,ncore)
#ectmp2 = -.25 * einsum('pqrs,pqrs', wpqrs[nocc:,nocc:,:ncore,:ncore]**2, dpqrs)
ectmp2_ccee = -2.0 * einsum('pqrs,pqrs', wpqrs[nocc:,nocc:,:ncore,:ncore]**2, dpqrs)
ectmp2_ccee += einsum('pqrs,pqsr,pqrs', wpqrs[nocc:,nocc:,:ncore,:ncore], wpqrs[nocc:,nocc:,:ncore,:ncore], dpqrs)
print "CT-MP2 Doubles Energy (CCEE):", ectmp2_ccee
    
# Core-to-active
dpq = e_bogo[ncore:nocc][:,None] + e_bogo[ncore:nocc]
drs = e_bogo[:ncore][:,None] + e_bogo[:ncore]
dpqrs = dpq.reshape(-1,1) + drs.reshape(-1)
dpqrs = 1.0/dpqrs.reshape(ncas,ncas,ncore,ncore)
#ectmp2 = -.25 * einsum('pqrs,pqrs', wpqrs[nocc:,nocc:,:ncore,:ncore]**2, dpqrs)
ectmp2_ccaa = -2.0 * einsum('pqrs,pqrs', wpqrs[ncore:nocc,ncore:nocc,:ncore,:ncore]**2, dpqrs)
ectmp2_ccaa += einsum('pqrs,pqsr,pqrs', wpqrs[ncore:nocc,ncore:nocc,:ncore,:ncore], wpqrs[ncore:nocc,ncore:nocc,:ncore,:ncore], dpqrs)
print "CT-MP2 Doubles Energy (CCAA):", ectmp2_ccaa
    
# Active-to-external
dpq = e_bogo[nocc:][:,None] + e_bogo[nocc:]
drs = e_bogo[ncore:nocc][:,None] + e_bogo[ncore:nocc]
dpqrs = dpq.reshape(-1,1) + drs.reshape(-1)
dpqrs = 1.0/dpqrs.reshape(nvir,nvir,ncas,ncas)
#ectmp2 = -.25 * einsum('pqrs,pqrs', wpqrs[nocc:,nocc:,:ncore,:ncore]**2, dpqrs)
ectmp2_aaee = -2.0 * einsum('pqrs,pqrs', wpqrs[nocc:,nocc:,ncore:nocc,ncore:nocc]**2, dpqrs)
ectmp2_aaee += einsum('pqrs,pqsr,pqrs', wpqrs[nocc:,nocc:,ncore:nocc,ncore:nocc], wpqrs[nocc:,nocc:,ncore:nocc,ncore:nocc], dpqrs)
print "CT-MP2 Doubles Energy (AAEE):", ectmp2_aaee

print "CT-MP2 Correlation Energy:   ", ectmp2_ccee + ectmp2_ccaa + ectmp2_aaee

e = emc + ectmp2_ccee + ectmp2_ccaa + ectmp2_aaee

print "CASSCF + CT-MP2 Total Energy:", e, "\n"
