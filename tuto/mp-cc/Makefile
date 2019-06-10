#!/usr/bin/env make

CC = gcc
FC = gfortran
LD = gfortran
CFLAGS = -O3 -mtune=native -fopenmp -fpic -lm
LFLAGS = -shared
FFLAGS = -O3 -mtune=native -fopenmp -fpic 

all: libmp2.so

OBJECTS = mp2.o dfmp2.o 

libmp2.so: $(OBJECTS)
	$(LD) $(LFLAGS) -o libmp2.so $(OBJECTS) -lblas

clean:
	/bin/rm *.so

mp2.o: mp2.h

.SUFFIXES:
.SUFFIXES: .c .o .f90

.f90.o:
	$(FC) -c $(FFLAGS) -o $@ $<

.c.o:
	$(CC) -c $(CFLAGS) -o $@ $<
