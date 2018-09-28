PROG = labo3
CFLAGS = -g -fPIC -m64 -Wall
LFLAGS= -fPIC -m64 -Wall
CC = mpicc

all: $(PROG)

labo3.o: labo3.c
	$(CC) $(CFLAGS) -c labo3.c

labo3 : labo3.o
	$(CC) $(LFLAGS) labo3.o -o labo3

run:
	mpirun --hostfile hostfile -np 24 labo3

ps:
	ps -fu $$USER

clean:
	/bin/rm -f *~
	/bin/rm -f *.o

test-mpi:
	mpirun --hostfile hostfile -np 24 labo3
