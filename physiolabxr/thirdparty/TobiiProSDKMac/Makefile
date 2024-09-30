#!/usr/bin/make -f 

.PHONY: all install

all:
	@echo "Nothing to build"

BITNESS?=64
PREFIX?=/usr/local/

TSDK_NAME:=tobii_research
install:
	@echo "Installing " $(TSDK_NAME)
	
	@mkdir -p $(DESTDIR)$(PREFIX)include/$(TSDK_NAME)
	@cp -v -r $(BITNESS)/include/* $(DESTDIR)$(PREFIX)include/$(TSDK_NAME)

	@mkdir -p $(DESTDIR)$(PREFIX)lib/$(TSDK_NAME)
	@cp -v -r $(BITNESS)/lib/* $(DESTDIR)$(PREFIX)lib/$(TSDK_NAME)
	@cp -v -r pkgconfig $(DESTDIR)$(PREFIX)lib

uninstall:
	@echo "Uninstalling " $(TSDK_NAME)
	
	@rm -v -r $(DESTDIR)$(PREFIX)include/$(TSDK_NAME)
	@rm -v -r $(DESTDIR)$(PREFIX)lib/$(TSDK_NAME)
	@rm -v  $(DESTDIR)$(PREFIX)lib/pkgconfig/$(TSDK_NAME).pc