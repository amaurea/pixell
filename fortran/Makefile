include ../manual.mk

all: _interpol_32.so _interpol_64.so _colorize.so _array_ops_32.so _array_ops_64.so

_%.so: %.f90
	FC="$(FC)" FFLAGS="$(FSAFE)" f2py --noopt -c -m _$(basename $<) $<  -lgomp skip: map_border calc_weights
	mv $(@:.so=.cpython-312-x86_64-linux-gnu.so) $@

%_32.f90: %.F90
	perl -pe 's/\b_\b/4/g' < $< > $@
%_64.f90: %.F90
	perl -pe 's/\b_\b/8/g' < $< > $@

clean:
	rm -rf *.pyc *.mod interpol*.f90 array_ops*.f90 *.so
