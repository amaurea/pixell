CPYFLAGS := -I. -shared -fPIC -fopenmp -lgomp -O3 -fdiagnostics-color=always -Wfatal-errors -lm
FPYFLAGS := -Wno-tabs -shared -fPIC -O3 -fopenmp -lgomp -fdiagnostics-color=always -Wfatal-errors
#CPYFLAGS := -I. -shared -fPIC -fopenmp -lgomp -g -O0 -fsanitize=address -fdiagnostics-color=always -Wfatal-errors -lm
#FPYFLAGS := -Wno-tabs -shared -fPIC -O0 -g -fsanitize=address -fopenmp -lgomp -fdiagnostics-color=always -Wfatal-errors

all: _cmisc.so _distances.so _srcsim.so _array_ops_32.so _array_ops_64.so _interpol_32.so _interpol_64.so _colorize.so clean_mod

_%.so: %.c
	gcc $(CPYFLAGS) -o $@ $^

_%.so: %.f90
	gfortran $(FPYFLAGS) -o $@ $^

%_32.f90: %.F90
	perl -pe 's/\b_\b/c_float/g' < $< > $@
%_64.f90: %.F90
	perl -pe 's/\b_\b/c_double/g' < $< > $@

clean: clean_mod
	rm -f *.so

clean_mod:
	rm -f *.mod

