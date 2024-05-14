.PHONY all: main.o
	/opt/mpich/bin/mpiexec -n 8 ./main.o

main.o: main.c
	/opt/mpich/bin/mpicc $^ -lm -o $@
