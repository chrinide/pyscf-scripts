#!/usr/bin/env python


#	WFs = 0
#	for k_id in range(self.num_kpts_loc): #self.num_kpts_loc
#		kpt = self.cell.get_abs_kpts(self.kpt_latt_loc[k_id])	
#		ao = numint.eval_ao(self.cell, grids_coor, kpt = kpt)
#		mo_included = numpy.dot(self.U[k_id], self.mo_coeff_kpts[k_id])[:,self.band_included_list]
#		mo_in_window = self.lwindow[k_id]
#		C_opt = mo_included[:,mo_in_window].dot(self.U_matrix_opt[k_id].T)
#		C_tildle = C_opt.dot(self.U_matrix[k_id].T)			
#		WFs = WFs + numpy.einsum('xi,in->xn', ao, C_tildle, optimize = True)
#		superWF = numpy.empty([nX, nY, nZ])
#		superWF_temp = numpy.empty([nX, nY, nZ])
#		WFs = self.get_wannier(grid = grid)
#		for wf_id in wf_list:
#			assert wf_id in list(range(self.num_wann_loc))
#			WF = WFs[:,wf_id].reshape(nx,ny,nz).real
#			for x in range(supercell[0]):
#				for y in range(supercell[1]):
#					for z in range(supercell[2]):					
#						superWF_temp[:nx,:ny,((nz-1)*z):((nz-1)*z + nz)] = WF
#					superWF_temp[:,((ny-1)*y):((ny-1)*y + ny),:] = superWF_temp[:,:ny,:]
#				superWF[((nx-1)*x):((nx-1)*x + nx),:,:] = superWF_temp[:nx,:,:]

#coord = mol.atom_coords()
#box = numpy.max(coord,axis=0) - numpy.min(coord,axis=0) + 6
#boxorig = numpy.min(coord,axis=0) - 3
# .../(nx-1) to get symmetric mesh
# see also the discussion on https://github.com/sunqm/pyscf/issues/154
#xs = numpy.arange(nx) * (box[0] / (nx - 1))
#ys = numpy.arange(ny) * (box[1] / (ny - 1))
#zs = numpy.arange(nz) * (box[2] / (nz - 1))
#coords = lib.cartesian_prod([xs,ys,zs])
#coords = numpy.asarray(coords, order='C') - (-boxorig)
#ngrids = nx * ny * nz
#blksize = min(8000, ngrids)
#rho = numpy.empty(ngrids)
#ao = None
#for ip0, ip1 in gen_grid.prange(0, ngrids, blksize):
#    ao = numint.eval_ao(mol, coords[ip0:ip1], out=ao)
#    rho[ip0:ip1] = numint.eval_rho(mol, ao, dm)
#rho = rho.reshape(nx,ny,nz)

#with open(outfile, 'w') as f:
#    f.write('Electron density in real space (e/Bohr^3)\n')
#    f.write('PySCF Version: %s  Date: %s\n' % (pyscf.__version__, time.ctime()))
#    f.write('%5d' % mol.natm)
#    f.write('%12.6f%12.6f%12.6f\n' % tuple(boxorig.tolist()))
#    f.write('%5d%12.6f%12.6f%12.6f\n' % (nx, xs[1], 0, 0))
#    f.write('%5d%12.6f%12.6f%12.6f\n' % (ny, 0, ys[1], 0))
#    f.write('%5d%12.6f%12.6f%12.6f\n' % (nz, 0, 0, zs[1]))
#    for ia in range(mol.natm):
#        chg = mol.atom_charge(ia)
#        f.write('%5d%12.6f'% (chg, chg))
#        f.write('%12.6f%12.6f%12.6f\n' % tuple(coord[ia]))
#
#    for ix in range(nx):
#        for iy in range(ny):
#            for iz in range(0,nz,6):
#                remainder  = (nz-iz)
#                if (remainder > 6 ):
#                    fmt = '%13.5E' * 6 + '\n'
#                    f.write(fmt % tuple(rho[ix,iy,iz:iz+6].tolist()))
#                else:
#                    fmt = '%13.5E' * remainder + '\n'
#                    f.write(fmt % tuple(rho[ix,iy,iz:iz+remainder].tolist()))
#                    break

