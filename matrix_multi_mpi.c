//
// Created by mouli on 3/10/16.
//
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define ROOT_PE 0
#define NPES 4
#define NR 10000
#define NC 10000

void initMatrix(float matrix[NR][NC]);

void printMatrix(float matrix[NR][NC]);

int main(int argc, char **argv) {
    float a[NR][NC];
    float b[NR][NC];
    float result[NR][NC];
    double start, end;
    int myId, numPE, from, to, i, j, k;
    MPI_Status mpiStatus;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myId);
    MPI_Comm_size(MPI_COMM_WORLD, &numPE);
    if (myId == ROOT_PE) {
        start = MPI_Wtime();
    }

    if (numPE != NPES) {
        if (myId == 0) {
            fprintf(stdout, "The example is only for %d PEs\n", NPES);
        }
        MPI_Finalize();
        exit(-1);
    }

    if (myId == ROOT_PE) {
        initMatrix(a);
        initMatrix(b);
    }

    from = myId * NC / numPE;
    to = (myId + 1) * NC / numPE;

    MPI_Bcast(b, NR * NC, MPI_FLOAT, ROOT_PE, MPI_COMM_WORLD);
    MPI_Scatter(a, NR * NC / numPE, MPI_FLOAT, a[from], NR * NC / numPE, MPI_FLOAT, ROOT_PE, MPI_COMM_WORLD);
//    printf("computing slice %d (from row %d to %d)\n", myId, from, to - 1);
    for (i = from; i < to; i++) {
        for (j = 0; j < NC; j++) {
            result[i][j] = 0;
            for (k = 0; k < NC; k++)
                result[i][j] += a[i][k] * b[k][j];
        }
    }

    MPI_Gather(result[from], NR * NC / numPE, MPI_FLOAT, result, NR * NC / numPE, MPI_FLOAT, ROOT_PE, MPI_COMM_WORLD);

    if (myId == ROOT_PE) {
        end = MPI_Wtime();
//        printMatrix(result);
        printf("Total time: %f \n", end - start);
    }
    MPI_Finalize();
    return 0;
}

void initMatrix(float matrix[NR][NC]) {
    int i, j;
    for (i = 0; i < NR; i++) {
        for (j = 0; j < NC; j++) {
            matrix[i][j] = i + j;
        }
    }
}

void printMatrix(float matrix[NR][NC]) {
    int i;
    int j;
    printf("##########Start printing matrix#################\n");
    for (i = 0; i < NR; i++) {
        for (j = 0; j < NC; j++) {
            printf("%.0f ", matrix[i][j]);
        }
        printf("\n");
    }
    printf("##########End printing matrix###################\n");
}
