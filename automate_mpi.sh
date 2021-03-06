#!/bin/bash 

for i in 200 400 600 800 1000 1200 1400 1600 1800 2000 2200 2400 2600 2800 3000 3200 3400 3600 3800 4000 4500 5000 5500 6000 7000 8000 9000 10000;
do
	find -type f -exec sed -i "s/\#define\ NR\ [0-9]*/\#define\ NR\ $i/g" {} \;
	find -type f -exec sed -i "s/\#define\ NC\ [0-9]*/\#define\ NC\ $i/g" {} \;
	mpicc -o mm_mpi matrix_multiplication_MPI.c;
	(echo -n "For size $i " && mpirun -np 4 ./mm_mpi) >> mpi_time_4
done
