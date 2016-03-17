#!/bin/bash 

for i in 200 400 600 800 1000 1200 1400 1600 1800 2000 2200 2400 2600 2800 3000 3200 3400 3600 3800 4000 4200 4400 4600 4800 5000 5200 5400 5600 5800 6000 6200 6400 6600 6800 7000 7200 7400 7600 7800 8000 8200 8400 8600 8800 9000 9200 9400 9600 9800;
do
	find -type f -exec sed -i "s/\#define\ NR\ [0-9]*/\#define\ NR\ $i/g" {} \;
	find -type f -exec sed -i "s/\#define\ NC\ [0-9]*/\#define\ NC\ $i/g" {} \;
	mpicc -o mm_mpi matrix_multiplication_MPI.c;
	(echo -n "For size $i " && mpirun -np 2 ./mm_mpi) >> mpi_time_2
done
