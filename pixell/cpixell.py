# This module implements a ctypes interface to the pixell C functions. This is
# a test to see if this is more convenient than cython or pure C extensions
import numpy as np, ctypes, os, platform
from . import bunch, utils

# Should probably split this into a ctypes_utils.py and individual files
# with the same interface as before, e.g. array_ops.py, interpol.py, etc.

if platform.uname()[0] == "Linux": default_ext = ".so"
else: default_ext = ".dylib"

def load(name, ext="auto"):
	"""Load shared library at given file name, supporting relative paths"""
	if ext == "auto": ext = default_ext
	return ctypes.CDLL(os.path.join(os.path.dirname(__file__),name + ext))


def declare(fun, argdesc, retdesc=None):
	fun.argtypes = parse_argdesc(argdesc)
	if retdesc is not None:
		fun.restype = parse_argdesc(retdesc)[0]

argmap = {
	"i":   ctypes.c_int,
	"i8":  ctypes.c_int8,
	"i16": ctypes.c_int16,
	"i32": ctypes.c_int32,
	"i64": ctypes.c_int64,
	"u":   ctypes.c_uint,
	"u8":  ctypes.c_uint8,
	"u16": ctypes.c_uint16,
	"u32": ctypes.c_uint32,
	"u64": ctypes.c_uint64,
	"f":   ctypes.c_float,
	"d":   ctypes.c_double,
}

def ptr_decomp(desc):
	toks = desc.split("*")
	if len(toks[0]) == 0 or any([len(t)>0 for t in toks[1:]]):
		raise ValueError("Invalid data type description. Must be non-empty and *'s can only occur at end")
	return toks[0], len(toks)-1

def parse_argdesc(desc):
	res  = []
	toks = desc.split(",")
	for tok in toks:
		name, nstar = ptr_decomp(tok)
		if nstar > 0: arg = ctypes.c_void_p
		else:         arg = argmap[name]
		res.append(arg)
	return res

def none_prepare(arrays):
	"""This function and none_finish make it easy to ignore Nones
	in a list of arrays. They work by making a None-less work array,
	and then reconstructing a None-full array at the end"""
	inds  = [i for i,a in enumerate(arrays) if a is not None]
	awork = [arrays[ind] for ind in inds]
	return awork, inds

def none_finish(awork, inds):
	# Make output list with the None's back in place
	oarrs = [None for a in inds]
	for i,ind in enumerate(inds):
		oarrs[ind] = awork[i]
	return oarrs

def harmonize_arrays(arrays, broadcast=False):
	awork, inds = none_prepare(arrays)
	awork = [np.asanyarray(a) for a in awork]
	dtype = np.result_type(*awork)
	awork = [a.astype(dtype, copy=False) for a in awork]
	if broadcast:
		awork = np.broadcast_arrays(*awork)
	return none_finish(awork, inds)

def listcontig(list_of_arrays, dtype=None):
	if isinstance(list_of_arrays, np.ndarray):
		return make_contig([list_of_arrays],dtype=dtype)[0]
	else:
		return [np.ascontiguousarray(a if dtype is None else a.astype(dtype,copy=False)) for a in list_of_arrays]

def make_contig(arrays, naxes=None, dtype=None):
	"""Make the last naxes axes of arrays contiguous. If naxes is 0,
	then all axes will be made contiguous. None-arrays are passed through
	unchanged"""
	awork, inds = none_prepare(arrays)
	for i,arr in enumerate(awork):
		arr = np.asanyarray(arr)
		if dtype is not None:
			awork[i] = arr.astype(dtype, copy=False)
		if not utils.iscontig(arr, naxes):
			awork[i] = np.ascontiguousarray(arr)
	return none_finish(awork, inds)

def lookup(dictionary, key):
	"""This is like indexing a dictionary, but uses the == operator
	to check each key. This is slower, but works for cases where ==
	has a looser equality criterion than normal key lookup"""
	for k,v in dictionary.items():
		if k == key: return v
	raise KeyError(key)

def ndoffs(shape, strides):
	inds = np.mgrid[tuple([slice(n) for n in shape])]
	offs = 0
	for i in range(len(shape)):
		offs += inds[i]*strides[i]
	return offs

def _ptrptr_helper(arr, naxes):
	if naxes == 0: return arr.ctypes.data, []
	else:
		# storage is there to keep all the ptr-arrays we construct
		# alive, otherwise they would be deallocated before we can use them
		ptrs, storage = zip(*[_ptrptr_helper(a, naxes-1) for a in arr])
		ptrs = np.array(ptrs,np.intp)
		return ptrs.ctypes.data, [ptrs,storage]

def _ptrptr_fast(arr, naxes):
	if naxes == 0: return arr.ctypes.data, []
	storage = []
	for i in range(naxes):
		nax = naxes-i
		arr = arr[(0,)*nax].ctypes.data + ndoffs(arr.shape[:nax],arr.strides[:nax])
		storage.append(arr)
	return arr.ctypes.data, storage

class ptrptr:
	def __init__(self, arr, naxes):
		if isinstance(arr, np.ndarray):
			self.data, self.storage = _ptrptr_fast(arr, naxes)
		else:
			self.data, self.storage = _ptrptr_helper(arr, naxes)

cmisc        = load("_cmisc")
distances    = load("_distances")
srcsim       = load("_srcsim")
array_ops_32 = load("_array_ops_32")
array_ops_64 = load("_array_ops_64")
interpol_32  = load("_interpol_32")
interpol_64  = load("_interpol_64")
_colorize    = load("_colorize")

declare(distances.free, "i*")

###########
# cmisc.c #
###########

declare(cmisc.alm2cl_sp, "i,i,i64*,f*,f*,f*")
declare(cmisc.alm2cl_dp, "i,i,i64*,d*,d*,d*")
declare(cmisc.transpose_alm_sp, "i,i,i64*,f*,f*")
declare(cmisc.transpose_alm_dp, "i,i,i64*,d*,d*")
declare(cmisc.lmul_sp, "i,i,i64*,f*,i,f*")
declare(cmisc.lmul_dp, "i,i,i64*,d*,i,d*")
declare(cmisc.lmatmul_sp, "i,i,i,i,i64*,f**,i,f**,f**")
declare(cmisc.lmatmul_dp, "i,i,i,i,i64*,d**,i,d**,d**")

# The goal of this module is to use python to automate some of the tedious
# wrapping work I had to do in the C extension (and cython). I want a simple
# way to describe what needs to be done to each argument, and have it happen
# pretty straightforwardly.
#
# Examples of things that may need to be done:
# * choose different functions given the data types of some of the arguments
# * repack some axes of Nd arrays into arrays of pointers
# * broadcast things
#
# But I think it's too ambitious to try to do all of this from a single description.
# Better to make convenience functions instead, and keep the C

#void alm2cl_sp(int lmax, int mmax, int64_t * mstart, float * alm1, float * alm2, float * cl) {


def alm2cl(ainfo, alm, alm2=None):
	"""Computes the cross power spectrum for the given alm and alm2, which
	must have the same dtype and broadcast. For example, to get the TEB,TEB
	cross spectra for a single map you would do
	 cl = ainfo.alm2cl(alm[:,None,:], alm[None,:,:])
	To get the same TEB,TEB spectra crossed with a different map it would
	be
	 cl = ainfo.alm2cl(alm1[:,None,:], alm2[None,:,:])
	In both these cases the output will be [{T,E,B},{T,E,B},nl]"""
	# alm2 can be None here. We carry that None all the way into the loop
	# below to avoid allocating more memory than necessary for the case where
	# alm is non-contiguous
	alm, alm2 = harmonize_arrays([alm, alm2], broadcast=True)
	mstart = np.ascontiguousarray(ainfo.mstart)
	core   = lookup({np.complex64:cmisc.alm2cl_sp, np.complex128:cmisc.alm2cl_dp}, alm.dtype)
	cl     = np.empty(alm.shape[:-1]+(ainfo.lmax+1,), utils.real_dtype(alm.dtype))
	# A common use case is to compute TEBxTEB auto-cross spectra, where
	# e.g. TE === ET since alm1 is the same array as alm2. To avoid duplicate
	# calculations in this case we use a cache, which skips computing the
	# cross-spectrum of any given pair of arrays more than once.
	cache  = {}
	for I in utils.nditer(alm.shape[:-1]):
			# Avoid duplicate calculation
		addr  = utils.getaddr(alm [I])
		addr2 = utils.getaddr(alm2[I]) if alm2 is not None else addr
		key   = tuple(sorted([addr,addr2]))
		if key in cache:
			cl[I] = cache[key]
		else:
			arow, arow2 = make_contig([alm[I], alm2[I]])
			if arow2 is None: arow2 = arow
			core(ainfo.lmax, ainfo.mmax, mstart.ctypes.data, arow.ctypes.data, arow2.ctypes.data, cl[I].ctypes.data)
			cache[key] = cl[I]
	return cl

def transpose_alm(ainfo, alm, out=None):
	"""In order to accomodate l-major ordering, which is not directoy
	supported by sharp, this function efficiently transposes Alm into
	Aml. If the out argument is specified, the transposed result will
	be written there. In order to perform an in-place transpose, call
	this function with the same array as "alm" and "out". If the out
	argument is not specified, then a new array will be constructed
	and returned."""
	alm = np.asanyarray(alm)
	if out is None:
		out = np.zeros_like(alm)
	work   = np.zeros(alm.shape[-1:], alm.dtype)
	mstart = np.ascontiguousarray(ainfo.mstart)
	core   = lookup({np.complex64:cmisc.transpose_alm_sp, np.complex128:cmisc.transpose_alm_dp}, alm.dtype)
	for I in utils.nditer(alm.shape[:-1]):
		arow = make_contig([alm[I]])[0]
		core(ainfo.lmax, ainfo.mmax, mstart.ctypes.data, arow.ctypes.data, work.ctypes.data)
		out[I] = work
	return out

def lmul(ainfo, alm, lfun, out=None):
	import warnings
	alm  = np.asarray(alm)
	rtype= utils.real_dtype(alm.dtype)
	if out is not None and (not utils.iscontig(out,naxes=1) or out.dtype != alm.dtype):
		raise ValueError("lmul's out argument must be contiguous along last axis, and have the same dtype as alm")
	mstart = np.ascontiguousarray(ainfo.mstart)
	# I should have made lmul and lmatmul different functions, but too late now I ugess
	if lfun.ndim == 3 and alm.ndim == 2:
		# Matrix multiplication
		if out is None: out = np.zeros(alm.shape, alm.dtype)
		core = lookup({np.complex64:cmisc.lmatmul_sp, np.complex128:cmisc.lmatmul_dp}, alm.dtype)
		awork, lwork = make_contig([alm, lfun.reshape(-1,lfun.shape[-1])], naxes=1)
		# Can't do this as part of the function call, as the ptrptr object is freed
		# before the function call happens
		aptr, lptr, optr = [ptrptr(a,1) for a in [awork, lwork, out]]
		core(lfun.shape[0], lfun.shape[1], ainfo.lmax, ainfo.mmax, mstart.ctypes.data,
				aptr.data, lfun.shape[-1]-1, lptr.data, optr.data)
	else:
		# Plain product
		# broadcast makes out unwritable
		alm, lfun = utils.broadcast_arrays(alm, lfun, npost=1)
		if out is None: out = np.zeros(alm.shape, alm.dtype)
		if out.shape != alm.shape:
			raise ValueError("lmul expected out to have shape %s, but got %s" % (str(alm.shape),str(out.shape)))
		core = lookup({np.complex64:cmisc.lmul_sp, np.complex128:cmisc.lmul_dp}, alm.dtype)
		for I in utils.nditer(alm.shape[:-1]):
			awork, lwork = make_contig([alm[I], lfun[I]])
			core(ainfo.lmax, ainfo.mmax, mstart.ctypes.data, awork.ctypes.data,
				lfun.shape[-1]-1, lwork.ctypes.data)
			out[I] = awork
	return out

# I have C versions of this, but this one is more flexible and should be fast enough,
# given that it's just copying things around
def transfer_alm(iainfo, ialm, oainfo, oalm=None, op=lambda a,b:b):
	"""Transfer alm from one layout to another."""
	if oalm is None:
		oalm = np.zeros(ialm.shape[:-1]+(oainfo.nelem,), ialm.dtype)
	lmax = min(iainfo.lmax, oainfo.lmax)
	mmax = min(iainfo.mmax, oainfo.mmax)
	if ialm.shape[:-1] != oalm.shape[:-1]:
		raise ValueError("ialm and oalm must agree on pre-dimensions")
	pshape = ialm.shape[:-1]
	npre = int(np.prod(pshape))
	# Numpy promotes uint64 to float64, so make an int64 view of mstart
	imstart = iainfo.mstart.view(np.int64)
	omstart = oainfo.mstart.view(np.int64)
	def transfer(dest, src, op): dest[:] = op(dest, src)
	for i in range(npre):
		I  = np.unravel_index(i, pshape)
		ia = ialm[I]; oa = oalm[I]
		for m in range(0, mmax+1):
			transfer(oa[omstart[m]+m*oainfo.stride:omstart[m]+(lmax+1)*oainfo.stride:oainfo.stride], ia[imstart[m]+m*iainfo.stride:imstart[m]+(lmax+1)*iainfo.stride:iainfo.stride], op)
	return oalm

############
# srcsim.c #
############

declare(srcsim.sim_objects, "i,f*,f*,i*,i*,f**,i,i*,i*,f**,f**,i,f,f,i,i,i,i,i,i,f*,f*,f**,f**,i,d*")
declare(srcsim.radial_sum, "i,f*,f*,i*,i*,i,f*,i,i,i,i,i,f*,f*,f**,f***,d*")

def sim_objects(map, obj_decs, obj_ras, obj_ys, obj_xs, amps, profs, prof_ids, posmap, vmin, rmax=0, separable=False, transpose=False, prof_equi=False, op="add", csize=8, return_times=False):
	"""
	map: array[ncomp,ny,nx]. Caller should make sure it has exactly 3 dims
	poss:   [{ra,dec},nobj] float
	pixs:   [{x,y},nobj] float
	amps:   [ncomp,nobj] float
	profs:  [nprof][{r,val}] float, can have different length for each profile
	prof_ids: [nprof] int
	posmap: [dec[ny],ra[nx]] if separable, otherwise [{dec,ra}][ny,nx]
	vmin:   The lowest value to evaluate profiles to. If this is set too low combined
	        with profiles that never reach zero, then things will get very slow.

	op:     How to combine contributions from multiple simulated objects and the input map.
	        add: Things add together linearly
	        max: Each pix will be the max of all contributions to that pixel
	        min: Each pix will be the min of all contributions to that pixel
	csize:   Size of cells used internally when determining which parts of the sky to
	         consider for each source, in pixels.
	
	Returns the resulting map. If inplace, then this will be the same object as the
	map that was passed in, otherwise it will be a new map.
	"""
	dtype  = np.float32
	ncomp  = len(map)
	if ncomp == 0: return map
	ny, nx = map[0].shape
	# Map must be contiguous along two last axes
	map = make_contig(map, dtype=dtype)
	pix_decs, pix_ras = make_contig([posmap[0],posmap[1]],dtype=dtype)
	obj_decs, obj_ras = make_contig([obj_decs, obj_ras],dtype=dtype)
	obj_ys, obj_xs, prof_ids = make_contig([obj_ys, obj_xs, prof_ids])
	amps = make_contig([np.asarray(amps)],1)[0]
	nobj = len(obj_ras)
	assert len(obj_decs) == len(obj_xs) == len(obj_ys) == nobj
	assert amps.shape[0] == ncomp and amps.shape[1] == nobj, "amps [%d,%d] must be [ncomp=%d,nobj=%d]" % (amps.shape[0], amps.shape[1], ncomp, nobj)
	# Set up the profiles. Too much looping in python here?
	nprof   = len(profs)
	prof_ns = np.array([len(p[0]) for p in profs],dtype=np.int32)
	prof_rs = make_contig([p[0] for p in profs],dtype=dtype)
	prof_vs = make_contig([p[1] for p in profs],dtype=dtype)
	# Pixel positions
	if separable:
		assert pix_decs.shape == (ny,) and pix_ras.shape == (nx,), "posmap ([%d],[%d]) must be [dec[ny=%d],ra[nx=%d]] if separable" % (len(pix_decs), len(pix_ras), ny, nx)
	else:
		assert pix_decs.shape == (ny,nx) and pix_ras.shape == (ny,nx), "posmap must be [{dec,ra},ny,nx] if not separable"
	# Timing
	times = np.zeros(3, dtype=np.float64)
	# Prepare to call
	iop = {"add":0, "max":1, "min":2}[op]
	amps_ptrs, prof_rs_ptrs, prof_vs_ptrs, map_ptrs = [ptrptr(a,1) for a in [amps, prof_rs, prof_vs, map]]
	# Yuck, what a long call
	srcsim.sim_objects(nobj, obj_decs.ctypes.data, obj_ras.ctypes.data,
		obj_ys.ctypes.data, obj_xs.ctypes.data, amps_ptrs.data,
		nprof, prof_ids.ctypes.data, prof_ns.ctypes.data, prof_rs_ptrs.data, prof_vs_ptrs.data,
		prof_equi, vmin, rmax, iop, ncomp, ny, nx, separable, transpose,
		pix_decs.ctypes.data, pix_ras.ctypes.data, map_ptrs.data, map_ptrs.data,
		csize, times.ctypes.data)
	if return_times: return map, times
	else:            return map

def radial_sum(map, obj_decs, obj_ras, obj_ys, obj_xs, rs, posmap, profs=None, separable=False, prof_equi=False, return_times=False):
	"""
	map: array[ncomp,ny,nx]. Caller should make sure it has exactly 3 dims
	poss:   [{ra,dec},nobj] float
	pixs:   [{x,y},nobj] float
	rs:     [nbin+1] float, bin edges (ascending). Faster if equi-spaced starting at 0
	posmap: [dec[ny],ra[nx]] if separable, otherwise [{dec,ra}][ny,nx]

	Returns the resulting profiles. If inplace, then this will be the same object as the
	map that was passed in, otherwise it will be a new map.
	"""
	dtype  = np.float32
	ncomp  = len(map)
	if ncomp == 0: return map
	ny, nx = map[0].shape
	# Map must be contiguous along two last axes
	map = make_contig(map, dtype=dtype)
	pix_decs, pix_ras     = make_contig([posmap[0],posmap[1]],dtype=dtype)
	obj_decs, obj_ras, rs = make_contig([obj_decs, obj_ras, rs],dtype=dtype)
	obj_ys,   obj_xs      = make_contig([obj_ys, obj_xs])
	nobj = len(obj_ras)
	assert len(obj_decs) == len(obj_xs) == len(obj_ys) == nobj
	# The profiles
	nbin = len(rs)-1
	assert rs.shape == (nbin+1,)
	if profs is None:
		profs = np.zeros((nobj,ncomp,nbin),dtype)
	elif not (utils.iscontig(profs,1) and profs.dtype == dtype):
		raise ValueError("Expected profs to be (%d,%s,%d) and %s" % (nobj,ncomp,nbin,str(dtype)))
	# Pixel positions
	if separable:
		assert pix_decs.shape == (ny,) and pix_ras.shape == (nx,), "posmap ([%d],[%d]) must be [dec[ny=%d],ra[nx=%d]] if separable" % (len(pix_decs), len(pix_ras), ny, nx)
	else:
		assert pix_decs.shape == (ny,nx) and pix_ras.shape == (ny,nx), "posmap must be [{dec,ra},ny,nx] if not separable"
	# Timing
	times = np.zeros(3, dtype=np.float64)
	# Prepare to call
	map_ptr   = ptrptr(map,  1)
	profs_ptr = ptrptr(profs,2)
	srcsim.radial_sum(nobj, obj_decs.ctypes.data, obj_ras.ctypes.data,
		obj_ys.ctypes.data, obj_xs.ctypes.data, nbin, rs.ctypes.data,
		prof_equi, ncomp, ny, nx, separable, pix_decs.ctypes.data, pix_ras.ctypes.data,
		map_ptr.data, profs_ptr.data, times.ctypes.data)
	if return_times: return profs, times
	else:            return profs

###############
# distances.c #
###############

class c_healpix_info(ctypes.Structure):
	_fields_ = [
		("nside", ctypes.c_int),
		("ny",    ctypes.c_int),
		("npix",  ctypes.c_int64),
		("ncap",  ctypes.c_int64),
		("nx",    ctypes.POINTER(ctypes.c_int)),
		("off",   ctypes.POINTER(ctypes.c_int64)),
		("shift", ctypes.POINTER(ctypes.c_int)),
		("ra0",   ctypes.POINTER(ctypes.c_double)),
		("dec",   ctypes.POINTER(ctypes.c_double)),
		("cos_dec", ctypes.POINTER(ctypes.c_double)),
		("sin_dec", ctypes.POINTER(ctypes.c_double)),
	]

declare(distances.distance_from_points_simple, "i64,d*,i64,d*,d*,i32*")
declare(distances.distance_from_points_simple_separable, "i64,i64,d*,d*,i64,d*,d*,i32*")
declare(distances.distance_from_points_bubble, "i,i,d*,i64,d*,i*,d,d*,i*")
declare(distances.distance_from_points_bubble_separable, "i,i,d*,d*,i64,d*,i*,d,d*,i*")
declare(distances.distance_from_points_cellgrid, "i,i,d*,d*,i64,d*,i*,i,i,d,d,i,d*,i*")
declare(distances.find_edges, "i64,i64,u8*,i64**", "i64")
declare(distances.find_edges_labeled, "i64,i64,i*,i64**", "i64")
# Healpix stuff
argmap["H"] = ctypes.POINTER(c_healpix_info)
declare(distances.build_healpix_info, "i", "H")
declare(distances.free_healpix_info, "H")
declare(distances.unravel_healpix, "H,i64,i64*,i32*")
declare(distances.ravel_healpix, "H,i64,i32*,i64*")
declare(distances.find_edges_healpix, "H,u8*,i**", "i64")
declare(distances.find_edges_labeled_healpix, "H,i32*,i**", "i64")
declare(distances.distance_from_points_bubble_healpix, "H,i64,d*,i*,d,d*,i*")
declare(distances.distance_from_points_heap_healpix, "H,i64,d*,i*,d,d*,i*")

def distance_from_points_simple(posmap, points, omap=None, odomains=None, domains=False, separable="auto"):
	"""distance_from_points(posmap, points, omap=None, odomains=None, domains=False)

	Given a posmap[{dec,ra},ny,nx] and a set of points[{dec,ra},npoint], computes the
	angular distance map from every pixel [ny,nx] to the nearest point. If domains==True,
	then a [ny,nx] map of the index of the nearest point is also returned. New arrays
	will be created for the output unless omap and/or odomains are specified, in which
	case they will be overwritten."""
	# Check that our inputs make sense
	H = _distances_setup(posmap, points, omap=omap, odomains=odomains, domains=domains, separable=separable)
	if not H.separable:
		distances.distance_from_points_simple(H.npix, H.posmap_ptr, H.npoint, H.points_ptr, H.omap_ptr, H.odomains_ptr)
	else:
		distances.distance_from_points_simple_separable(H.ny, H.nx, H.ypos_ptr, H.xpos_ptr, H.npoint, H.points_ptr, H.omap_ptr, H.odomains_ptr)
	if domains: return H.omap, H.odomains
	else: return H.omap

def distance_from_points_simple_separable(ypos, xpos, points, omap=None, odomains=None, domains=False):
	"""distance_from_points_simple_separable(ypos, xpos, points, omap=None, odomains=None, domains=False)

	Like distance_from_points, but optimized for the case where the coordinate system
	is separable, as is typically the case for cylindrical projections. Instead of a full
	posmap[{dec,ra},ny,nx] it takes ypos[ny] which gives the dec of each point along the y axis
	and xpos[nx] which gives the ra of each point along the x axis. The main advantage of this
	is that one can avoid the somewhat heavy computation of the full posmap."""
	return distance_from_points_simple([ypos,xpos], points, omap=omap, odomains=odomains, domains=domains, separable=True)

def distance_from_points_bubble(posmap, point_pos, point_pix, rmax=None, omap=None,
	odomains=None, domains=False, separable="auto"):
	# Must set up domains even if we don't want them, since the C code assumes they're allocated
	H = _distances_setup(posmap, point_pos, omap=omap, odomains=odomains, domains=True, point_pix=point_pix, separable=separable)
	assert H.point_pix is not None
	rmax = 0 if rmax is None else rmax
	if not H.separable:
		distances.distance_from_points_bubble(H.ny, H.nx, H.posmap_ptr, H.npoint, H.points_ptr, H.point_pix_ptr, rmax, H.omap_ptr, H.odomains_ptr)
	else:
		distances.distance_from_points_bubble_separable(H.ny, H.nx, H.ypos_ptr, H.xpos_ptr, H.npoint, H.points_ptr, H.point_pix_ptr, rmax, H.omap_ptr, H.odomains_ptr)
	if domains: return H.omap, H.odomains
	else: return H.omap

def distance_from_points_bubble_separable(ypos, xpos, point_pos, point_pix, rmax=None, omap=None, odomains=None, domains=False):
	"""Like distance_from_points_bubble, but optimized for the case where the coordinate system
	is separable, as is typically the case for cylindrical projections. Instead of a full
	posmap[{dec,ra},ny,nx] it takes ypos[ny] which gives the dec of each point along the y axis
	and xpos[nx] which gives the ra of each point along the x axis. The main advantage of this
	is that one can avoid the somewhat heavy computation of the full posmap. If rmax is specified,
	the calculation will stop at the distance rmax. Beyond that distance domain will be set to -1 and
	distance will be set to rmax. This can be used to speed up the calculation if one only cares about
	distances up to a certain point."""
	return distance_from_points_bubble([ypos,xpos], point_pos, point_pix, rmax=rmax, omap=omap,
		odomains=odomains, domains=domains, separable=True)

def distance_from_points_cellgrid(ypos, xpos, point_pos, point_pix, dr=np.inf, rmax=None, omap=None, odomains=None, domains=False, bsize=32):
	"""dr = np.inf and bsize=32 are based on benchmarking on my laptop. Reducing dr was never
	beneficial and could damage performance significantly if set too low. bsize in the range
	10-50 are OK, but worst-case performance rapidly gets worse after that. The best bsize was
	slightly dependent on the size of the map"""
	# Must set up domains even if we don't want them, since the C code assumes they're allocated
	H = _distances_setup([ypos,xpos], point_pos, omap=omap, odomains=odomains, domains=True, point_pix=point_pix, separable="auto", allow_posmap_list=True)
	assert H.point_pix is not None
	rmax  = 0 if rmax is None else rmax
	bsize = np.zeros(2,np.int32)+bsize
	distances.distance_from_points_cellgrid(H.ny, H.nx, H.ypos_ptr, H.xpos_ptr, H.npoint, H.points_ptr, H.point_pix_ptr, bsize[0], bsize[1], rmax, dr, H.separable, H.omap_ptr, H.odomains_ptr)
	if domains: return H.omap, H.odomains
	else:       return H.omap

def find_edges(mask, flat=False):
	"""find_edges(mask, flat=False)

	Given a 2d numpy array mask[ny,nx], returns an list of indices (y[:],x[:]) for each
	pixel at the edge of a zero region in the mask. If flat==True then it will instead
	return a list of indicees [:] into the flattened mask."""
	# Ensure that we have the right data type and contiguity. Try extra hard to avoid copies
	# when the input is known to have a compatible data type.
	mask = make_contig([mask],np.uint8)
	assert mask.ndim == 2, "mask must be 2D"
	edges_ptr = ctypes.POINTER(ctypes.c_int64)()
	n = distances.find_edges(mask.shape[0], mask.shape[1], mask.ctypes.data, ctypes.byref(edges_ptr))
	# Copy into numpy array
	if n > 0:
		edges = np.ctypeslib.as_array(edges_ptr, (n,)).copy()
	else:
		edges = np.zeros(0, np.int64)
	distances.free(edges_ptr)
	if not flat:
		edges = np.unravel_index(edges, mask.shape)
	return edges

def find_edges_labeled(labels, flat=False):
	"""find_edges_labeled(labels, flat=False)

	Given a 2d numpy array labels[ny,nx], returns an list of indices (y[:],x[:]) for each
	pixel at the edge of a region with constant, nonzero value in labels. If flat==True then it will instead
	return a list of indicees [:] into the flattened labels."""
	# Ensure that we have the right data type and contiguity. Try extra hard to avoid copies
	# when the input is known to have a compatible data type.
	labels = make_contig([labels],dtype=np.int32)
	assert labels.ndim == 2, "labels must be 2D"
	edges_ptr = ctypes.POINTER(ctypes.c_int64)()
	n = distances.find_edges_labeled(labels.shape[0], labels.shape[1], labels.ctypes.data, ctypes.byref(edges_ptr))
	# Copy into numpy array
	if n > 0:
		edges = np.ctypeslib.as_array(edges_ptr, (n,)).copy()
	else:
		edges = np.zeros(0, np.int64)
	distances.free(edges_ptr)
	if not flat:
		edges = np.unravel_index(edges, mask.shape)
	return edges

#### healpix stuff below here ####

class healpix_info:
	def __init__(self, nside):
		self.ptr  = distances.build_healpix_info(nside)
		self.info = self.ptr.contents
	def __dealloc__(self):
		distances.free_healpix_info(self.ptr)
	@property
	def nside(self):   return self.info.nside
	@property
	def npix(self):    return self.info.npix
	@property
	def ny(self):      return self.info.ny
	@property
	def nx(self):      return np.ctypeslib.as_array(self.info.nx, (self.info.ny,))
	@property
	def off(self):     return np.ctypeslib.as_array(self.info.off, (self.info.ny,))
	@property
	def shift(self):   return np.ctypeslib.as_array(self.info.shift, (self.info.ny,))
	@property
	def ra0(self):     return np.ctypeslib.as_array(self.info.ra0, (self.info.ny,))
	@property
	def dec(self):     return np.ctypeslib.as_array(self.info.dec, (self.info.ny,))
	@property
	def cos_dec(self): return np.ctypeslib.as_array(self.info.cos_dec, (self.info.ny,))
	@property
	def sin_dec(self): return np.ctypeslib.as_array(self.info.sin_dec, (self.info.ny,))

def find_edges_healpix(info, mask, flat=True):
	"""find_edges_healpix(healpix_info info, mask, flat=True)

	Given a healpix_info info and a boolean mask[:], returns
	the pixel indices of all the pixels at the edge of the zero
	region of the mask. If flat=False, then the indices will be
	returned as [{row,col},:]. The default is to return them as
	plain 1d indices."""
	# Flat is the default for healpix because healpix is usually 1d
	make = make_contig([mask],dtype=np.int8)[0]
	assert mask.ndim == 1, "mask must be 1D"
	edges_ptr = ctypes.POINTER(ctypes.c_int)()
	nedge = distances.find_edges_healpix(info.ptr, mask.ctypes.data, ctypes.byref(edges_ptr))
	# Could avoid this copy if numpy would assume ownership...
	edges = np.ctypeslib.as_array(edges_ptr, (2,nedge)).copy()
	distances.free(edges_ptr)
	if flat:
		edges = info.off[edges[0]]+edges[1]
	return edges

def find_edges_labeled_healpix(info, labels, flat=True):
	"""find_edges_labeled_healpix(healpix_info info, labels, flat=True)

	Given a healpix_info info and integer labels[:], returns
	the pixel indices of all the pixels at the the edge of a
	region with the same, nonzero value in labels. If flat=False,
	then the indices will be returned as [{row,col},:]. The default
	is to return them as plain 1d indices."""
	labels = make_contig([labels],dtype=np.int32)[0]
	assert labels.ndim == 1, "labels must be 1D"
	edges_ptr = ctypes.POINTER(ctypes.c_int)()
	nedge = distances.find_edges_labeled_healpix(info.ptr, labels.ctypes.data, ctypes.byref(edges_ptr))
	edges = np.ctypeslib.as_array(edges_ptr, (2,nedge)).copy()
	if flat:
		edges = info.off[edges[0]]+edges[1]
	return edges

def distance_from_points_healpix(info, point_pos, point_pix, rmax=None, omap=None, odomains=None, domains=False, method="bubble"):
	"""distance_from_points_healpix(healpix_info info, point_pos, point_pix, rmax=None, omap=None, odomains=None, domains=False)

	Computes the distance of each healpix pixel in the sky described by healpix_info info to
	the closest of the points with position point_pos[{dec,ra},npoint] and corresponding
	pixel position point_pix[{y,x},npoint]. If omap is specified, the result is written there.
	In any case the result is also returned. If domains=True then a domain map is also returned.
	In this map the value of each pixel is the index of the closest point in the point_pos array,
	or -1 if no valid point was found. The main way this could happen is if rmax is specified,
	which makes the search stop after a distance of rmax radians. Distances larger than this
	will be capped to this value."""
	# Check that our inputs make sense
	point_pos = make_contig([point_pos],dtype=np.float64)[0]
	assert point_pos.ndim == 2 and len(point_pos) == 2, "point_pos must be [{dec,ra},npoint]"
	point_pix = make_contig([point_pix],dtype=np.int32)[0]
	# point_pix can be standard 1d or our internal 2d format
	if point_pix.ndim == 1:
		point_pix = unravel_healpix(info, point_pix)
	assert point_pix.ndim == 2 and len(point_pix) == 2 and point_pix.shape[1] == point_pos.shape[1], "point_pos must be [npoint] or [{y,x},npoint]"
	if omap is None: omap = np.empty(info.npix, dtype=np.float64)
	assert utils.iscontig(omap)
	assert omap.ndim == 1 and omap.size == info.npix and omap.dtype==np.float64, "omap must be [npix] float64"
	if odomains is None: odomains = np.empty(info.npix, dtype=np.int32)
	assert odomains.ndim == 1 and odomains.size == info.npix and odomains.dtype==np.int32, "odomains must be [npix] int32"
	assert utils.iscontig(odomains)
	fun = lookup({"bubble":distances.distance_from_points_bubble_healpix, "heap":distances.distance_from_points_heap_healpix}, method)
	fun(info.ptr, point_pos.shape[1], point_pos.ctypes.data, point_pix.ctypes.data, rmax or 0.0, omap.ctypes.data, odomains.ctypes.data)
	if domains: return omap, odomains
	else:       return omap

def unravel_healpix(info, pix1d):
	pix1d = make_contig([pix1d],dtype=np.int64)[0]
	assert pix1d.ndim == 1
	pix2d = np.zeros((2,)+pix1d.shape, np.int32)
	distances.unravel_healpix(info.ptr, pix1d.size, pix1d.ctypes.data, pix2d.ctypes.data)
	return pix2d

def ravel_healpix(info, pix2d):
	pix2d = make_contig([pix2d],dtype=np.int32)[0]
	pix1d = np.zeros(pix2d.shape[1:], np.int64)
	distances.ravel_healpix(info.ptr, pix1d.size, pix2d.ctypes.data, pix1d.ctypes.data)
	return pix1d

# Pure python helpers

def _distances_setup(posmap, points, omap=None, odomains=None, domains=False,
		point_pix=None, dtype=np.float64, separable="auto", allow_posmap_list=False):
	if separable == "auto": separable = posmap[0].ndim == 1
	if separable:
		assert posmap[0].ndim == posmap[1].ndim == 1, "ypos and xpos must be 1d with lengths ny and nx"
		posmap = make_contig([posmap[0],posmap[1]],dtype=dtype)
		ny, nx = posmap[0].size, posmap[1].size
	else:
		assert posmap[0].ndim == 2 and posmap[0].shape == posmap[1].shape, "posmap must have shape [{dec,ra},ny,nx]"
		if allow_posmap_list:
			# distance_from_points_cellgrid needs this
			posmap = make_contig(posmap, dtype=dtype)
		else:
			posmap = make_contig([posmap], dtype=dtype)[0]
		ny, nx = posmap[0].shape
	points = make_contig([points],dtype=dtype)[0]
	if point_pix is not None:
		point_pix = make_contig([point_pix],dtype=np.int32)[0]
		assert point_pix.shape == points.shape
		if separable:
			point_pix = fix_point_pix_separable(posmap[0], posmap[1], points, point_pix)
		else:
			point_pix = fix_point_pix(posmap, points, point_pix)
	if omap is None: omap = np.empty_like(posmap[0], shape=(ny,nx),dtype=dtype)
	assert omap.ndim == 2 and omap.shape[-2:] == (ny,nx) and omap.dtype==dtype, "omap must be [ny,nx] float64"
	if domains:
		if odomains is None: odomains = np.empty_like(posmap[0], shape=(ny,nx), dtype=np.int32)
		assert odomains.ndim == 2 and odomains.shape[-2:] == (ny,nx) and odomains.dtype==np.int32, "odomains must be [ny,nx] int32"
	return bunch.Bunch(
			ny=ny, nx=nx, npix=ny*nx, npoint=points.shape[1], separable=separable,
			omap=omap, omap_ptr=omap.ctypes.data,
			odomains=odomains, odomains_ptr=odomains.ctypes.data if domains else None,
			points=points, points_ptr=points.ctypes.data,
			point_pix=point_pix, point_pix_ptr=point_pix.ctypes.data if point_pix is not None else None,
			posmap=posmap, posmap_ptr=posmap.ctypes.data if isinstance(posmap, np.ndarray) else None,
			ypos=posmap[0], ypos_ptr=posmap[0].ctypes.data,
			xpos=posmap[1], xpos_ptr=posmap[1].ctypes.data)

def fix_point_pix(posmap, point_pos, point_pix):
	"""fix_point_pix(posmap, point_pos, point_pix)

	Return a new point_pix where out of bounds points have been replaced by the closest point
	on the boundary. This uses the simple method for the pixels on the boundary. The number of pixels
	on the boundary are much lower than the pixels in the main part of the image, which partially
	makes up for the slowness of the simple method, but you can still expect slowness if too many points
	are outside the image."""
	from scipy import ndimage
	ypos, xpos = posmap
	# Get the bad points
	bad = np.where(np.any(point_pix < 0,0) | np.any(point_pix >= np.array(ypos.shape[-2:])[:,None],0))[0]
	# If there aren't any, we can just return right away
	nbad = len(bad)
	if nbad == 0: return point_pix
	# Otherwise, run the simple method on each boundary. The simple method is slow, but the boundary is
	# has much fewer pixels.
	point_bad = point_pos[:,bad]
	pos_edges = np.array([np.concatenate([p[:,0],p[:,-1],p[0,:],p[-1,:]],-1) for p in [ypos,xpos]])
	dist, dom = distance_from_points_simple(pos_edges[:,None,:], point_bad, domains=True)
	dist, dom = dist[0], dom[0]
	# Find the minimum position for each point
	minpos1d = np.array(ndimage.minimum_position(dist, dom+1, np.arange(nbad)+1)).reshape(-1)
	# Turn the 1d minpos into a 2d pixel position
	minpos   = _unwrap_minpos(minpos1d, ypos.shape[0], ypos.shape[1])
	# Copy these into the output point_pix
	opoint_pix = point_pix.copy()
	opoint_pix[:,bad] = minpos
	return opoint_pix

def fix_point_pix_separable(ypos, xpos, point_pos, point_pix):
	"""fix_point_pix_separable(ypos, xpos, point_pos, point_pix)

	Return a new point_pix where out of bounds points have been replaced by the closest point
	on the boundary. This uses the simple method for the pixels on the boundary. The number of pixels
	on the boundary are much lower than the pixels in the main part of the image, which partially
	makes up for the slowness of the simple method, but you can still expect slowness if too many points
	are outside the image."""
	from scipy import ndimage
	# Get the bad points
	bad = np.where(np.any(point_pix < 0,0) | (point_pix[0] >= ypos.size) | (point_pix[1] >= xpos.size))[0]
	# If there aren't any, we can just return right away
	nbad = len(bad)
	if nbad == 0: return point_pix
	# Otherwise, run the simple method on each boundary. The simple method is slow, but the boundary is
	# has much fewer pixels.
	point_bad = point_pos[:,bad]
	pos_edges = np.array([
		np.concatenate([ypos, ypos, np.full(xpos.size, ypos[0]), np.full(xpos.size, ypos[-1])]),
		np.concatenate([np.full(ypos.size, xpos[0]), np.full(ypos.size, xpos[-1]), xpos, xpos])])
	dist, dom = distance_from_points_simple(pos_edges[:,None,:], point_bad, domains=True)
	dist, dom = dist[0], dom[0]
	# Find the minimum position for each point
	minpos1d = np.array(ndimage.minimum_position(dist, dom+1, np.arange(nbad)+1)).reshape(-1)
	# Turn the 1d minpos into a 2d pixel position
	minpos   = _unwrap_minpos(minpos1d, ypos.size, xpos.size)
	# Copy these into the output point_pix
	opoint_pix = point_pix.copy()
	opoint_pix[:,bad] = minpos
	return opoint_pix

def _unwrap_minpos(minpos1d, ny, nx):
	minpos   = np.zeros([2,len(minpos1d)], minpos1d.dtype)
	mask = minpos1d < ny;
	minpos[0,mask], minpos[1,mask] = minpos1d[mask], 0
	mask = (minpos1d >= ny) & (minpos1d < 2*ny)
	minpos[0,mask], minpos[1,mask] = minpos1d[mask]-ny, nx-1
	mask = (minpos1d >= 2*ny) & (minpos1d < 2*ny+nx)
	minpos[0,mask], minpos[1,mask] = 0, minpos1d[mask]-2*ny
	mask = minpos1d >= 2*ny+nx
	minpos[0,mask], minpos[1,mask] = ny-1, minpos1d[mask]-(2*ny+nx)
	return minpos

#############
# array_ops #
#############

# This old fortran stuff is some of the first low-level code I
# wrote for enlib. It's pretty inefficient and should probably
# be replaced with a newer version

declare(array_ops_32.matmul_multi_sym, "f*,f*,i,i,i")
declare(array_ops_64.matmul_multi_sym, "d*,d*,i,i,i")
declare(array_ops_32.matmul_multi, "f*,f*,f*,i,i,i,i")
declare(array_ops_64.matmul_multi, "d*,d*,f*,i,i,i,i")
declare(array_ops_32.ang2rect, "f*,f*,i")
declare(array_ops_64.ang2rect, "d*,d*,i")
declare(array_ops_32.find_contours, "f*,f*,i*,i,i,i")
declare(array_ops_64.find_contours, "d*,d*,i*,i,i,i")
declare(array_ops_32.roll_rows, "f*,i*,f*,i,i")
declare(array_ops_64.roll_rows, "d*,i*,d*,i,i")

# This is is pretty clunky, and can probably be optimized a lot.
# It should be faster to loop over the pixel dimension first
# Also, the symmetric version is slower than the other
# Still, despite these things the function is fast enough to
# live with for now
# A[npix,n,m] b[npix,k,m] → out[npix,k,n]
def matmul(A, b, axes=[-2,-1], sym=False):
	# Prepare our arrays. This is surprisingly involved!
	# The overhead probably is probably quite high too
	A = np.asanyarray(A)
	b = np.asanyarray(b)
	dtype = np.result_type(A,b)
	if dtype != np.float32 and dtype != np.float64:
		raise ValueError("Only float32 and float64 supported")
	axes = [ax%A.ndim for ax in axes]
	bax  = axes[:len(axes)-(A.ndim-b.ndim)]
	Af   = utils.partial_flatten(A, axes)
	bf   = utils.partial_flatten(b, bax)
	if len(Af) != len(bf):
		raise ValueError("A and b must agree on the number of elements")
	add_dim = bf.ndim == 2
	if add_dim: bf = bf[:,None,:]
	Af, bf = make_contig([Af,bf],dtype=dtype)
	npix, n, m = Af.shape
	k = bf.shape[1]
	# We're now ready to call our function
	if sym:
		outf= bf.copy()
		vals= [ctypes.c_int(a) for a in [npix,n,k]]
		fun = lookup({np.float32:array_ops_32.matmul_multi_sym, np.float64:array_ops_64.matmul_multi_sym},dtype)
		fun(Af.ctypes.data, outf.ctypes.data, *vals)
	else:
		outf= np.zeros((npix,k,m),dtype)
		vals= [ctypes.c_int(a) for a in [npix,n,m,k]]
		fun = lookup({np.float32:array_ops_32.matmul_multi, np.float64:array_ops_64.matmul_multi},dtype)
		fun(Af.ctypes.data, bf.ctypes.data, outf.ctypes.data, *vals)
	# Reshape our output array back to the right shape
	if add_dim: outf = outf[:,0,:]
	out = utils.partial_expand(outf, b.shape, bax)
	return out

def matmul_sym(A, b, axes=[-2,-1]):
	return matmul(A, b, axes=axes, sym=True)

def ang2rect(angs):
	"""angs[n,2]→rect[n,3]"""
	angs = np.ascontiguousarray(angs)
	fun  = lookup({np.float32:array_ops_32.ang2rect, np.float64:array_ops_64.ang2rect}, angs.dtype)
	rect = np.zeros((len(angs),3),dtype=angs.dtype)
	fun(angs.ctypes.data, rect.ctypes.data, len(angs))
	return rect

def find_contours(imap, vals, omap=None):
	imap, vals = make_contig([imap,vals])
	assert imap.ndim == 2
	vals = vals.astype(imap.dtype, copy=False)
	if omap is None: omap = np.zeros_like(imap, dtype=np.int32)
	assert omap.shape == imap.shape and omap.dtype == np.int32
	fun  = lookup({np.float32:array_ops_32.find_contours, np.float64:array_ops_64.find_contours}, imap.dtype)
	fun(imap.ctypes.data, vals.ctypes.data, omap.ctypes.data, imap.shape[0], imap.shape[1], len(vals))
	return omap

def roll_rows(imap, offsets, omap=None):
	imap, offsets = make_contig([imap,offsets])
	offsets = offsets.astype(np.int32, copy=False)
	assert imap.ndim == 2
	assert offsets.shape == imap.shape[:1]
	if omap is None: omap = np.zeros_like(imap)
	assert omap.shape == imap.shape and omap.dtype == imap.dtype
	fun  = lookup({np.float32:array_ops_32.roll_rows, np.float64:array_ops_64.roll_rows}, imap.dtype)
	fun(imap.ctypes.data, offsets.ctypes.data, omap.ctypes.data, imap.shape[0], imap.shape[1])
	return omap

############
# interpol #
############

declare(interpol_32.interpol, "f*,i*,f*,f*,i,i,i,i,i,i,i,i")
declare(interpol_64.interpol, "d*,i*,d*,d*,i,i,i,i,i,i,i,i")
declare(interpol_32.interpol_deriv, "f*,i*,f*,f*,i,i,i,i,i,i,i,i")
declare(interpol_64.interpol_deriv, "d*,i*,d*,d*,i,i,i,i,i,i,i,i")
declare(interpol_32.spline_filter1d, "f*,i*,i,i,i,i,i,i")
declare(interpol_64.spline_filter1d, "d*,i*,i,i,i,i,i,i")

def map_coordinates(idata, points, odata=None, mode="spline", order=3, border="cyclic", trans=False, deriv=False,
		prefilter=True):
	"""An alternative implementation of scipy.ndimage.map_coordinates. It is slightly
	slower (20-30%), but more general. Basic usage is
	 odata[{pre},{pdims}] = map_coordinates(idata[{pre},{dims}], points[ndim,{pdims}])
	where {foo} means a (possibly empty) shape. For example, if idata has shape (10,20)
	and points has shape (2,100), then the result will have shape (100,), and if
	idata has shape (10,20,30,40) and points has shape (3,1,2,3,4), then the result
	will have shape (10,1,2,3,4). Except for the presence of {pre}, this is the same
	as how map_coordinates works.

	It is also possible to pass the output array as an argument (odata), which must
	have the same data type as idata in that case.

	The function differs from ndimage in the meaning of the optional arguments.
	mode specifies the interpolation scheme to use: "conv", "spline" or "lanczos".
	"conv" is polynomial convolution, which is commonly used in image processing.
	"spline" is spline interpolation, which is what ndimage uses.
	"lanczos" convolutes with a lanczos kernerl, which approximates the optimal
	sinc kernel. This is slow, and the quality is not much better than spline.

	order specifies the interpolation order, its exact meaning differs based on
	mode.

	border specifies the handling of boundary conditions. It can be "zero",
	"nearest", "cyclic" or "mirror"/"reflect". The latter corresponds to ndimage's
	"reflect". The others do not match ndimage due to ndimage's inconsistent
	treatment of boundary conditions in spline_filter vs. map_coordiantes.

	trans specifies whether to perform the transpose operation or not.
	The interpolation performed by map_coordinates is a linear operation,
	and can hence be expressed as out = A*data, where A is a matrix.
	If trans is true, then what will instead be performed is data = A.T*in.
	For this to work, the odata argument must be specified. This will be
	read from, while idata will be written to.

	Normally idata is read and odata is written to, but when trans=True,
	idata is written to and odata is read from.

	If deriv is True, then the function will compute the derivative of the
	interpolation operation with respect to the position, resulting in
	odata[ndim,{pre},{pdims}]
	"""

	imode   = {"conv":0, "spline":1, "lanczos":2}[mode]
	iborder = {"zero":0, "nearest":1, "cyclic":2, "mirror":3, "reflect":3}[border]
	idata, points = make_contig([idata,points])
	dtype = idata.dtype
	if dtype != np.float32 and dtype != np.float64: dtype = points.dtype
	if dtype != np.float32 and dtype != np.float64: dtype = np.float64
	idata, points = [a.astype(dtype,copy=False) for a in [idata, points]]
	ndim    = points.shape[0]
	dpre,dpost= idata.shape[:-ndim], idata.shape[-ndim:]
	dpost   = np.array(dpost, np.int32)
	def iprod(x): return np.prod(x).astype(int)
	npre, ngrid, nsamp = [iprod(a) for a in [dpre,dpost,points.shape[1:]]]
	if not trans:
		if not deriv: oshape = dpre+points.shape[1:]
		else:         oshape = (ndim,)+dpre+points.shape[1:]
		if odata is None: odata = np.empty(oshape, dtype=dtype)
		assert odata.dtype == dtype
		assert odata.shape == oshape
		assert utils.iscontig(odata)
		if mode == "spline" and prefilter:
			idata = spline_filter(idata, order=order, border=border, ndim=ndim, trans=False)
		iflat = idata.reshape(npre,ngrid)
		pflat = points.reshape(ndim,nsamp)
		if not deriv:
			fun = lookup({np.float32:interpol_32.interpol, np.float64:interpol_64.interpol}, idata.dtype)
			oflat = odata.reshape(npre,nsamp)
			fun(iflat.ctypes.data, dpost.ctypes.data, oflat.ctypes.data, pflat.ctypes.data,
				npre, ndim, ngrid, nsamp, imode, order, iborder, 0)
		else:
			fun = lookup({np.float32:interpol_32.interpol_deriv, np.float64:interpol_64.interpol_deriv}, idata.dtype)
			oflat = odata.reshape(ndim,npre,nsamp)
			fun(iflat.ctypes.data, dpost.ctypes.data, oflat.ctypes.data, pflat.ctypes.data,
				npre, ndim, ngrid, nsamp, imode, order, iborder, 0)
		return odata
	else:
		# We cannot infer the shape of idata from odata and points. So both
		# idata and odata must be specified in this case.
		iflat = idata.reshape(npre,ngrid)
		pflat = points.reshape(ndim,nsamp)
		if not deriv:
			fun = lookup({np.float32:interpol_32.interpol, np.float64:interpol_64.interpol}, idata.dtype)
			oflat = odata.reshape(npre,nsamp)
			fun(iflat.ctypes.data, dpost.ctypes.data, oflat.ctypes.data, pflat.ctypes.data,
				npre, ndim, ngrid, nsamp, imode, order, iborder, 1)
		else:
			fun = lookup({np.float32:interpol_32.interpol_deriv, np.float64:interpol_64.interpol_deriv}, idata.dtype)
			oflat = odata.reshape(ndim,npre,nsamp)
			fun(iflat.ctypes.data, dpost.ctypes.data, oflat.ctypes.data, pflat.ctypes.data,
				npre, ndim, ngrid, nsamp, imode, order, iborder, 1)
		if mode == "spline" and prefilter:
			idata[:] = spline_filter(idata, order=order, border=border, ndim=ndim, trans=True)
		return idata

def spline_filter(data, order=3, border="cyclic", ndim=None, trans=False):
	"""Apply a spline filter to the given array. This is normally done on-the-fly
	internally in map_coordinates when using spline interpolation of order > 1,
	but since it's an operation that applies to the whole input array, it can be
	a big overhead to do this for every call if only a small number of points are
	to be interpolated. This overhead can be avoided by manually filtering the array
	once, and then passing in the filtered array to map_coordinates with prefilter=False
	to turn off the internal filtering."""
	data = np.asanyarray(data).copy()
	assert utils.iscontig(data)
	iborder = {"zero":0, "nearest":1, "cyclic":2, "mirror":3}[border]
	if ndim is None: ndim = data.ndim
	dshape  = np.array(data.shape, np.int32)
	fun = lookup({np.float32:interpol_32.spline_filter1d, np.float64:interpol_64.spline_filter1d}, data.dtype)
	for axis in range(data.ndim-ndim,data.ndim)[::-1 if trans else 1]:
		fun(data.ctypes.data, dshape.ctypes.data, axis, data.ndim, data.size, order, iborder, trans)
	return data

############
# colorize #
############

declare(_colorize.remap, "d*,i16*,d*,i16*,i,i,i")
declare(_colorize.direct,"d*,i16*,i,i")
declare(_colorize.direct_colorcap,"d*,i16*,i,i")

def colorize_scalar_fortran(a, desc):
	a, vals = make_contig([a,desc.vals],dtype=np.float64)
	cols    = make_contig([desc.cols],  dtype=np.int16)[0]
	res     = np.empty((len(a),4),dtype=np.int16)
	_colorize.remap(a.ctypes.data, res.ctypes.data, vals.ctypes.data, cols.ctypes.data, len(a), len(vals), 4)
	return res.astype(np.uint8)

def colorize_direct_fortran(a, desc):
	a   = make_contig([a],dtype=np.float64)[0]
	res = np.empty((a.shape[1],4),dtype=np.uint16)
	_colorize.direct(a.ctypes.data, res.ctypes.data, a.shape[1], a.shape[0])
	return res.astype(np.uint8)

def colorize_direct_colorcap_fortran(a, desc):
	a   = make_contig([a],dtype=np.float64)[0]
	res = np.empty((a.shape[1],4),dtype=np.uint16)
	_colorize.direct_colorcap(a.ctypes.data, res.ctypes.data, a.shape[1], a.shape[0])
	return res.astype(np.uint8)
