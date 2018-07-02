CC = gcc
CFLAGS = -O3 -fopenmp -fpic -lm -shared

LD = gfortran
LDFLAGS = -shared

all: libvv10.so

libvv10.so: vv10.c
	$(CC) $(CFLAGS) -o $@ $^

clean:
	/bin/rm *.o *.so
