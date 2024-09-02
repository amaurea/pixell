from __future__ import print_function
import numpy as np, time
from . import utils
from .cpixell import map_coordinates, spline_filter

###### The functions below deal with building multilinear interpolations for arbitrary functions ####

def build(func, interpolator, box, errlim, maxsize=None, maxtime=None, return_obox=False, return_status=False, verbose=False, nstart=None, *args, **kwargs):
	"""Given a function func([nin,...]) => [nout,...] and
	an interpolator class interpolator(box,[nout,...]),
	(where the input array is regularly spaced in each direction),
	which provides __call__([nin,...]) => [nout,...],
	automatically polls func and constructs an interpolator
	object that has the required accuracy inside the provided
	bounding box."""
	box     = np.asfarray(box)
	errlim  = np.asfarray(errlim)
	idim    = box.shape[1]
	n       = [3]*idim if nstart is None else nstart
	n       = np.array(n) # starting mesh size
	x       = utils.grid(box, n)
	obox    = [np.inf,-np.inf]

	t0      = time.time()

	# Set up initial interpolation
	ip = interpolator(box, func(x), *args, **kwargs)

	errs = [np.inf]*idim # Max error for each *input* dimension in last refinement step
	err  = np.max(errs)
	depth = 0
	# Refine until good enough
	while True:
		depth += 1
		if maxsize and np.prod(n) > maxsize:
			if return_status:
				return ip if not return_obox else ip, np.array(obox), False, err
			raise OverflowError("Maximum refinement mesh size exceeded")
		nok = 0
		# Consider accuracy for each input parameter by tentatively doubling
		# resolution for that parameter and checking how well the interpolator
		# predicts the true values.
		for i in range(idim):
			if any(errs[i] > errlim):
				if maxtime and time.time() - t0 > maxtime:
					if return_status:
						return ip if not return_obox else ip, np.array(obox), False, err
					raise OverflowError("Maximum refinement time exceeded")
				# Grid may not be good enough in this direction.
				# Try doubling resolution
				nnew   = n.copy()
				nnew[i]= nnew[i]*2+1
				x      = utils.grid(box, nnew)
				# These have shape [ndim,dim1,dim2,dim3,...]
				yinter = ip(x)
				ytrue  = func(x)
				if np.any(np.isnan(ytrue)):
					raise ValueError("Function to interpolate returned invalid value")
				err = np.max(np.abs((ytrue-yinter).reshape(ytrue.shape[0],-1)), 1)
				# Find the worst failure:
				ytrue_flat = ytrue.reshape(4,-1)
				yinter_flat= yinter.reshape(4,-1)
				badi = np.argmax(np.abs((ytrue_flat[2]-yinter_flat[2])))
				if verbose: print(x.shape, x.size, err/errlim)
				if any(err > errlim):
					# Not good enough, so accept improvement
					ip = interpolator(box, ytrue, *args, **kwargs)
					n  = nnew
				else: nok += 1
				errs[i] = err
				# update output box
				obox[0] = np.minimum(obox[0], np.min(ytrue.reshape(ytrue.shape[0],-1),1))
				obox[1] = np.maximum(obox[0], np.max(ytrue.reshape(ytrue.shape[0],-1),1))
			else: nok += 1
		if nok >= idim: break
	res = (ip,)
	if return_obox: res = res + (np.array(obox),)
	if return_status: res = res + (True,err)
	return res[0] if len(res) == 1 else res

class Interpolator:
	def __init__(self, box, y, *args, **kwargs):
		self.box, self.y = np.array(box), np.array(y)
		self.args, self.kwargs = args, kwargs

class ip_ndimage(Interpolator):
	def __call__(self, x):
		ix = ((x.T-self.box[0])/(self.box[1]-self.box[0])*(np.array(self.y.shape[1:])-1)).T
		return utils.interpol(self.y, ix, *self.args, **self.kwargs)

class ip_linear(Interpolator):
	# General bilinear interpolation. This does the same as ndimage interpolation
	# using order=1, but is about 3 times slower.
	def __init__(self, box, y, *args, **kwargs):
		Interpolator.__init__(self, box, y, *args, **kwargs)
		self.n, self.npre = self.box.shape[1], y.ndim-self.box.shape[1]
		self.ys = lin_derivs_forward(y, self.npre)
	def __call__(self, x):
		flatx = x.reshape(x.shape[0],-1)
		# Get the float cell index of each sample
		px = ((flatx.T-self.box[0])/(self.box[1]-self.box[0])*(np.array(self.ys.shape[-self.n:]))).T
		ix = (np.floor(px)).astype(int)
		ix = np.maximum(0,np.minimum(np.array(self.ys.shape[-self.n:])[:,None]-1,ix))
		fx = px-ix
		res = np.zeros(self.ys.shape[self.n:self.n+self.npre]+fx.shape[1:2])
		for i in range(2**self.n):
			I = np.unravel_index(i,(2,)*self.n)
			res += self.ys[I][(slice(None),)*self.npre+tuple(ix)]*np.prod(fx**(np.array(I)[:,None]),0)
		return res.reshape(res.shape[:-1]+x.shape[1:])

class ip_grad(Interpolator):
	"""Gradient interpolation. Faster but less accurate than bilinear"""
	def __init__(self, box, y, *args, **kwargs):
		Interpolator.__init__(self, box, y, *args, **kwargs)
		self.n, self.npre = self.box.shape[1], y.ndim-self.box.shape[1]
		self.ys  = lin_derivs_forward(y, self.npre)
	def __call__(self, x):
		flatx = x.reshape(x.shape[0],-1)
		px = ((flatx.T-self.box[0])/(self.box[1]-self.box[0])*np.array(self.ys.shape[-self.n:])).T
		ix = (np.floor(px)).astype(int)
		ix = np.maximum(0,np.minimum(np.array(self.ys.shape[-self.n:])[:,None]-1,ix))
		fx = px-ix
		res = np.zeros(self.ys.shape[self.n:self.n+self.npre]+fx.shape[1:2])
		inds = np.concatenate([np.zeros(self.n,dtype=int)[None], np.eye(self.n,dtype=int)],0)
		for I in inds:
			res += self.ys[tuple(I)][(slice(None),)*self.npre+tuple(ix)]*np.prod(fx**(np.array(I)[:,None]),0)
		return res.reshape(res.shape[:-1]+x.shape[1:])

#class ip_grad(Interpolator):
#	def __init__(self, box, y, *args, **kwargs):
#		y = np.asarray(y)
#		self.box = np.array(box)
#		self.n, self.npre = self.box.shape[1], y.ndim-self.box.shape[1]
#		self.dy  = grad_forward(y, self.npre)
#		self.y   = y[(slice(None,-1),)*y.nsim]
#	def __call__(self, x):
#		flatx = x.reshape(x.shape[0],-1)
#		px = ((flatx.T-self.box[0])/(self.box[1]-self.box[0])*np.array(self.ys.shape[-self.n:])).T
#		ix = (np.floor(px)).astype(int)
#		ix = np.maximum(0,np.minimum(np.array(self.ys.shape[-self.n:])[:,None]-1,ix))
#		fx = px-ix
#		res = self.y[tuple(ix)]
#		for i in range(self.n):
#			res += self.dy[tuple(ix)]*fx[i]
#		return res.reshape(res.shape[:-1]+x.shape[1:])

def lin_derivs_forward(y, npre=0):
	"""Given an array y with npre leading dimensions and n following dimensions,
	compute all combinations of the 0th and 1st derivatives along the n last
	dimensions, returning an array of shape (2,)*n+(:,)*npre+(:-1,)*n. That is,
	it is one shorter in each direction along which the derivative is taken.
	Derivatives are computed using forward difference."""
	y        = np.asfarray(y)
	nin      = y.ndim-npre
	ys = np.zeros((2,)*nin+y.shape)
	ys[(0,)*nin] = y
	for i in range(nin):
		whole,start,end = slice(None,None,None), slice(0,-1,None), slice(1,None,None)
		target = (whole,)*(i)+(1,)+(0,)*(nin-i-1)
		source = (whole,)*(i)+(0,)+(0,)*(nin-i-1)
		cells1 = (whole,)*(npre+i)+(start,)+(whole,)*(nin-i-1)
		cells2 = (whole,)*(npre+i)+(end,)  +(whole,)*(nin-i-1)
		ys[target+cells1] = ys[source+cells2]-ys[source+cells1]
	ys = ys[(slice(None),)*(nin+npre)+(slice(0,-1),)*nin]
	return ys

def grad_forward(y, npre=0):
	"""Given an array y with npre leading dimensions and n following dimensions,
	the gradient along the n last dimensions, returning an array of shape (n,)+y.shape.
	Derivatives are computed using forward difference."""
	y        = np.asfarray(y)
	nin      = y.ndim-npre
	dy       = np.zeros((nin,)+y.shape)
	for i in range(nin):
		whole,start,end = slice(None,None,None), slice(0,-1,None), slice(1,None,None)
		source = (whole,)*i+(0,)+(0,)*(nin-i-1)
		cells1 = (whole,)*(npre+i)+(start,)+(whole,)*(nin-i-1)
		cells2 = (whole,)*(npre+i)+(end,)  +(whole,)*(nin-i-1)
		dy[i][cells1] = y[source+cells2]-y[source+cells1]
	return dy[(slice(None),)+(slice(None,-1),)*(dy.ndim-1)]
