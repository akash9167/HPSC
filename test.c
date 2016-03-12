#include<stdio.h>
#include<stdlib.h>
#include<mpi.h>

#define NPES 1
#define NR 1000
#define NC 1000
//int NR, NC;	
void initialize(float a[NR][NC]);
void print(float c[NR][NC]);

int main(int argc, char** argv){
	
//	scanf("%d",&NR);
//	NC = NR;
	float a[NR][NC];
	float b[NR][NC];
	float c[NR][NC];

	double start, end, total;
	int rank, 
	    npes, 
		i, 
		j, 
		k,
		from,
		to;

	//start = MPI_Wtime();
/////////////////////////////////////////////////////////////////////////
/*	//printf("Start:%f \n", start);
	initialize(a);	
	initialize(b);
	
	for(i=0; i<NC; i++){
		for(j=0; j<NR; j++){
			c[i][j] = 0;
			for(k=0; k<NC; k++) c[i][j] += a[i][k]*b[j][k];
		}
	}
	//print(c);
	end = MPI_Wtime();
	//printf("End:%f \n", end);
	printf("Total time taken is: %f\n",end-start);*/
/////////////////////////////////////////////////////////////////////////
	
	MPI_Init(NULL, NULL);
	MPI_Comm_size(MPI_COMM_WORLD, &npes);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	
	
	if(rank == 0){
		start = MPI_Wtime();
		//printf("Start:%f \n", start);
		printf("NPES: %d\n",npes);
		if(npes!=NPES){
			//printf("YO1\n");
			fprintf(stderr,"This program is for %d Processes.",NPES);
			MPI_Finalize();
			exit(-1);
		}
		else{
			//printf("YO\n");
			initialize(a);	
			initialize(b);
		}
	}
	
	MPI_Bcast(b, NR * NC, MPI_FLOAT, 0, MPI_COMM_WORLD);
	
	from = rank*NR/npes;
	to = (rank+1)*NR/npes;

	MPI_Scatter(a, (to-from)*NC, MPI_FLOAT, a[from], (to-from)*NC, MPI_FLOAT, 0, MPI_COMM_WORLD);

	for(i=from; i<to; i++){
		for(j=0; j<NR; j++){
			c[i][j] = 0;
			for(k=0; k<NC; k++) c[i][j] += a[i][k]*b[j][k];
		}
	}
	
	MPI_Gather(c[from], (to-from)*NC, MPI_FLOAT, c, (to-from)*NC, MPI_FLOAT, 0, MPI_COMM_WORLD);

	if(rank == 0){
		//print(c);
		end = MPI_Wtime();
		total = end - start;
		printf("Total time taken is: %f\n",total);	
	}
	//printf("End:%f \n", end);
	MPI_Finalize();
	
	return 0;
}

void initialize(float a[NR][NC]){
	int i,j;
	for(i=0; i<NR; i++){
		for(j=0; j<NC; j++){
			a[i][j] = i+j;		
		}
	}
}

void print(float c[NR][NC]){
	int i, j;
	for (i = 0; i < NR; i++) {
    	for (j = 0; j < NC; j++) {
            printf("%0.f\t", c[i][j]);
        }
        printf("\n");
    }
}
