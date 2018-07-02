#!/usr/bin/env python

import numpy
from pyscf import lib
from functools import reduce
einsum = lib.einsum

def getdffno(mf,ncore,eri_mo,thresh_vir=1e-4):

  lib.logger.info(mf,"* FNO orbital construction ")
  lib.logger.info(mf,"* VIR threshold: %s" % thresh_vir)

  mol = mf.mol
  nao, nmo = mf.mo_coeff.shape
  nocc = mol.nelectron//2 - ncore
  nvir = nmo - nocc - ncore
  mo_core = mf.mo_coeff[:,:ncore]
  mo_occ = mf.mo_coeff[:,ncore:ncore+nocc]
  mo_vir = mf.mo_coeff[:,ncore+nocc:]

  co = mo_occ
  cv = mo_vir
  eo = mf.mo_energy[ncore:ncore+nocc]
  ev = mf.mo_energy[ncore+nocc:]

  lib.logger.info(mf,"* VIRxVIR block of MP2 matrix")
  vv_denom = -ev.reshape(-1,1)-ev
  dab = numpy.zeros((len(ev),len(ev)))
  for i in range(nocc):
      eps_i = eo[i]
      i_Qv = eri_mo[:, i, :].copy()
      for j in range(nocc):
          eps_j = eo[j]
          j_Qv = eri_mo[:, j, :].copy()
          div = 1.0/(eps_i + eps_j + vv_denom)
          t2ij = div*einsum('Qa,Qb->ab', i_Qv, j_Qv)
          dab += 2.0*einsum('ab,cb->ac', t2ij,t2ij) 
          dab -= 1.0*einsum('ab,bc->ac', t2ij,t2ij) 
  dab = 2.0*dab      
  dab = (dab+dab.T)*0.5
  natoccvir, natorbvir = numpy.linalg.eigh(-dab)
  for i, k in enumerate(numpy.argmax(abs(natorbvir), axis=0)):
      if natorbvir[k,i] < 0:
          natorbvir[:,i] *= -1
  natoccvir = -natoccvir
  lib.logger.debug(mf,"* Occupancies")
  lib.logger.debug(mf,"* %s" % natoccvir)
  lib.logger.debug(mf,"* The sum is %8.6f" % numpy.sum(natoccvir)) 
  active = (thresh_vir <= natoccvir)
  lib.logger.info(mf,"* Natural Orbital selection")
  nvir = cv.shape[1]
  for i in range(nvir):
      lib.logger.debug(mf,"orb: %d %s %8.6f" % (i,active[i],natoccvir[i]))
      actIndices = numpy.where(active)[0]
  lib.logger.info(mf,"* Original active orbitals %d" % len(ev))
  lib.logger.info(mf,"* New active orbitals %d" % len(actIndices))
  lib.logger.debug(mf,"* Active orbital indices %s" % actIndices)
  lib.logger.info(mf,"* Virtual core orbitals: %d" % (len(ev)-len(actIndices)))
  natorbvir = natorbvir[:,actIndices]                                    
  fvv = numpy.diag(ev)
  fvv = reduce(numpy.dot, (natorbvir.T, fvv, natorbvir))
  fnoe, fnov = numpy.linalg.eigh(fvv)
  cv = reduce(numpy.dot,(cv,natorbvir,fnov))
  ev = fnoe
  nvir = len(actIndices)

  return cv,ev 

def getfno(mf,ncore,thresh_vir=1e-4):

  lib.logger.info(mf,"* FNO orbital construction ")
  lib.logger.info(mf,"* VIR threshold: %s" % thresh_vir)

  mol = mf.mol
  nao, nmo = mf.mo_coeff.shape
  nocc = mol.nelectron//2 - ncore
  nvir = nmo - nocc - ncore
  mo_core = mf.mo_coeff[:,:ncore]
  mo_occ = mf.mo_coeff[:,ncore:ncore+nocc]
  mo_vir = mf.mo_coeff[:,ncore+nocc:]

  co = mo_occ
  cv = mo_vir
  eo = mf.mo_energy[ncore:ncore+nocc]
  ev = mf.mo_energy[ncore+nocc:]
  e_denom = 1.0/(eo.reshape(-1, 1, 1, 1) - \
            ev.reshape(-1, 1, 1) + eo.reshape(-1, 1) - ev)

  lib.logger.info(mf,"* Building amplitudes")
  eri_mo = ao2mo.general(mf._eri, (co,cv,co,cv), compact=False)
  eri_mo = eri_mo.reshape(nocc,nvir,nocc,nvir)
  t2amp = eri_mo * e_denom

  # Virtual part
  lib.logger.info(mf,"* VIRxVIR block of MP2 matrix")
  dab = 2.0*einsum('iajb,icjb->ac', t2amp, t2amp) - \
        1.0*einsum('iajb,ibjc->ac', t2amp, t2amp)    
  dab = 2.0*dab      
  dab = (dab+dab.T)*0.5
  natoccvir, natorbvir = numpy.linalg.eigh(-dab)
  for i, k in enumerate(numpy.argmax(abs(natorbvir), axis=0)):
      if natorbvir[k,i] < 0:
          natorbvir[:,i] *= -1
  natoccvir = -natoccvir
  lib.logger.debug(mf,"* Occupancies")
  lib.logger.debug(mf,"* %s" % natoccvir)
  lib.logger.debug(mf,"* The sum is %8.6f" % numpy.sum(natoccvir)) 
  active = (thresh_vir <= natoccvir)
  lib.logger.info(mf,"* Natural Orbital selection")
  nvir = cv.shape[1]
  for i in range(nvir):
      lib.logger.debug(mf,"orb: %d %s %8.6f" % (i,active[i],natoccvir[i]))
      actIndices = numpy.where(active)[0]
  lib.logger.info(mf,"* Original active orbitals %d" % len(ev))
  lib.logger.info(mf,"* New active orbitals %d" % len(actIndices))
  lib.logger.debug(mf,"* Active orbital indices %s" % actIndices)
  lib.logger.info(mf,"* Virtual core orbitals: %d" % (len(ev)-len(actIndices)))
  natorbvir = natorbvir[:,actIndices]                                    
  fvv = numpy.diag(ev)
  fvv = reduce(numpy.dot, (natorbvir.T, fvv, natorbvir))
  fnoe, fnov = numpy.linalg.eigh(fvv)
  cv = reduce(numpy.dot,(cv,natorbvir,fnov))
  ev = fnoe
  nvir = len(actIndices)

  return cv,ev 

