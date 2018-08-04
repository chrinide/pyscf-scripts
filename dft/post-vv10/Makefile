CC = gcc
CFLAGS = -O3 -mtune=native -fopenmp -fpic -lm -shared

all: libvv10.so

libvv10.so: vv10.c
	$(CC) $(CFLAGS) -o $@ $^

clean:
	/bin/rm *.so
