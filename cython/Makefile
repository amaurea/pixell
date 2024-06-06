
include ../manual.mk

all: cmisc.so distances.so srcsim.so

cmisc.so: cmisc.c cmisc_core.c
	gcc $(CPYFLAGS) -o $@ $^

srcsim.so: srcsim.c srcsim_core.c
	gcc $(CPYFLAGS) -o $@ $^

distances.so: distances.c distances_core.c distances_core.h
	gcc $(CPYFLAGS) -o $@ $^

%.so: %.c
	gcc $(CPYFLAGS) -o $@ $^

%.c: %.pyx
	cython --fast-fail -o $@ $<
	perl -pi -e 's/typedef npy_float64 _Complex/typedef double _Complex/; s/typedef npy_float32 _Complex/typedef float _Complex/' $@

cmisc.c: cmisc.pyx cmisc.pxd cmisc_core.h cmisc_core.c

distances.c: distances.pyx distances_wrap.pxd distances_core.h distances_core.c

srcsim.c: srcsim.pyx srcsim_wrap.pxd srcsim_core.h srcsim_core.c

clean:
	rm -f cmisc.c distances.c srcsim.c cmisc.so distances.so srcsim.so
