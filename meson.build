project('pixell', ['c','fortran'])
py = import('python').find_installation(pure: false)

# Dependencies
omp_dep = dependency('openmp')

# Libraries
cc = meson.get_compiler('c')
c_m_dep = cc.find_library('m', required: true)

fc = meson.get_compiler('fortran')
add_project_arguments('-Wno-tabs', language : 'fortran')
add_project_arguments('-Wno-conversion', language : 'fortran')

# Directories
odir = py.get_install_dir() / 'pixell'

# We make one shared library per file for now
fortran_sources = {
	'compiled/colorize.f90': '_colorize',
	'compiled/interpol_32.f90': '_interpol_32',
	'compiled/interpol_64.f90': '_interpol_64',
	'compiled/array_ops_32.f90': '_array_ops_32',
	'compiled/array_ops_64.f90': '_array_ops_64',
}

foreach srcs, name : fortran_sources
	shared_library(name, srcs, dependencies: [omp_dep],
		install: true, install_dir: odir, name_prefix: '')
endforeach

c_sources = {
	'compiled/cmisc.c': '_cmisc',
	'compiled/srcsim.c': '_srcsim',
	'compiled/distances.c': '_distances',
}

foreach srcs, name : c_sources
	shared_library(name, srcs, dependencies: [omp_dep, c_m_dep],
		install: true, install_dir: odir, name_prefix: '')
endforeach

# The python sources. Must be manually updated when new files are added.
# Meson supports globbing, but heavily discourages doing so due to what
# appears to be technical limitations, where meson needs to be reinvoked
# if new files are added, and if one forgets this, one will get strange
# errors. But really, with the manual approach one will also get strange
# errors if one forgets to update this build script, so it seems exactly
# the same to me.
python_sources = [
	'pixell/__init__.py',
	'pixell/aberration.py',
	'pixell/analysis.py',
	'pixell/array_ops.py',
	'pixell/bunch.py',
	'pixell/cgrid.py',
	'pixell/colorize.py',
	'pixell/colors.py',
	'pixell/coordinates.py',
	'pixell/curvedsky.py',
	'pixell/enmap.py',
	'pixell/enplot.py',
	'pixell/fft.py',
	'pixell/interpol.py',
	'pixell/lensing.py',
	'pixell/memory.py',
	'pixell/mpi.py',
	'pixell/multimap.py',
	'pixell/pointsrcs.py',
	'pixell/powspec.py',
	'pixell/reproject.py',
	'pixell/resample.py',
	'pixell/tilemap.py',
	'pixell/uharm.py',
	'pixell/utils.py',
	'pixell/wavelets.py',
	'pixell/wcsutils.py',
	# Not technically a python source file, but is required for pixell
	'pixell/arial.ttf',
]

py.install_sources(
	python_sources,
	subdir: 'pixell',
)

python_scripts = [
	'scripts/benchmark_pixell_runner.py',
	'scripts/benchmark_pixell.py',
]

py.install_sources(
	python_scripts,
	subdir: 'pixell/scripts',
)
