#!/bin/bash 

for i in 6000 6500 7000 8000 9000 9800;
do
	find -type f -exec sed -i "s/\#define\ Row\ [0-9]*/\#define\ Row\ $i/g" {} \;
	find -type f -exec sed -i "s/\#define\ Col\ [0-9]*/\#define\ Col\ $i/g" {} \;
	mpicc -o omp_mm -fopenmp matrix_mult_OPENMP.c;
	(echo -n "For size $i " && ./omp_mm) >> openmp_result_2_2
done
