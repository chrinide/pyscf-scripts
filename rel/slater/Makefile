#!/usr/bin/env make

CC = gcc
LD = gcc
CDEBUG = -Wpedantic -g -pg -Wunused -Wall
CFLAGS = -O3 -mtune=native -fpic -lm -fopenmp #$(CDEBUG)
LFLAGS = -shared

all: libci.so

OBJECTS = ci.o 

libci.so: $(OBJECTS)
	$(LD) $(LFLAGS) -o libci.so $(OBJECTS)

clean:
	/bin/rm -f libci.so $(OBJECTS)

ci.o: ci.h

.SUFFIXES:
.SUFFIXES: .c .o

.c.o:
	$(CC) -c $(CFLAGS) -o $@ $<

