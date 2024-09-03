# Debug settings. Included address sanitation, so you may need something like
# ASAN_OPTIONS=detect_leaks=0 LD_PRELOAD=/lib/libasan.so python ... when testing
#CPYFLAGS := -I. -shared -fPIC -fopenmp -lgomp -g -O0 -fsanitize=address -fdiagnostics-color=always -Wfatal-errors -lm
#FPYFLAGS := -Wno-tabs -shared -fPIC -O0 -g -fsanitize=address -fopenmp -lgomp -fdiagnostics-color=always -Wfatal-errors
# Standard optimized compile
CPYFLAGS := -I. -shared -fPIC -fopenmp -lgomp -O3 -fdiagnostics-color=always -Wfatal-errors -lm
FPYFLAGS := -Wno-tabs -shared -fPIC -O3 -fopenmp -lgomp -fdiagnostics-color=always -Wfatal-errors

all: _cmisc.so _distances.so _srcsim.so _array_ops_32.so _array_ops_64.so _interpol_32.so _interpol_64.so _colorize.so

_%.so: %.c
	gcc $(CPYFLAGS) -o $@ $^

_%.so: %.f90
	gfortran $(FPYFLAGS) -o $@ $^
	rm -f $*.mod

%_32.f90: %.F90
	perl -pe 's/\b_\b/c_float/g;s/@@/32/g' < $< > $@
%_64.f90: %.F90
	perl -pe 's/\b_\b/c_double/g;s/@@/64/g' < $< > $@

clean: clean_mod
	rm -f *.so

clean_mod:
	rm -f *.mod
