CFLAGS=-g -fPIC -m64 -Wall
LFLAGS=-fPIC -m64 -Wall -framework opencl
CC=clang

all: TP4

TP4.o: TP4.c
	$(CC) $(CFLAGS) -c TP4.c -o $@

TP4: TP4.o
	$(CC) $(LFLAGS) TP4.o -o TP4

clean:
	rm TP4
	rm TP4.o
