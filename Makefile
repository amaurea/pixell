# Standard meson build
SHELL=/bin/bash
# Editable install. May break in an externally managed environment.
# If you have trouble with this, consider the inline build below
editable:
	python -m pip install --no-build-isolation --editable .
# do DESTDIR=... make install to override where things are installed
install: build
	(cd _build; meson install)
test: inline
	pytest
build: _build
	(cd _build; meson compile)
# Inline build. Does not actually install anything, but you can
# get an editable build by creating a symlink to pixell/pixell
# somewhere in pythonpath
inline: build
	(shopt -s nullglob; cd pixell; rm -f *.so; ln -s ../_build/*.so ../_build/*.dylib .)
# Manual build. Does not use meson. Controlled by compiled/Makefile.
# Not general, but easy to debug and adapt to own system
manual: unlink
	$(MAKE) -C compiled all
	(cd pixell && ln -s ../compiled/*.so .)
clean: unlink
	if [ -d _build ]; then (cd _build; ninja clean); fi
	$(MAKE) -C compiled clean
distclean: unlink
	rm -r _build

# Helpers
_build:
	mkdir -p _build
	meson setup _build
unlink:
	(shopt -s nullglob; rm -f pixell/*.so pixell/*.dylib)
