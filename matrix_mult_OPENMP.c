#include<stdio.h>
#include<stdlib.h>
#include<omp.h>
#include<mpi.h>

#define THREADS 2
#define Row 9800
#define Col 9800


int main(){
	int a[Row][Col],
		b[Row][Col],
		c[Row][Col],
		chunk=10,
		i,j,k,tmp, tid, nthreads;
	double start, end;
	start = MPI_Wtime();
	omp_set_dynamic(0);
	omp_set_num_threads(2);
	
	#pragma omp parallel shared(a,b,c,chunk) private(i,j,k)
	{
		tid = omp_get_thread_num();
		if(tid==0) nthreads = omp_get_num_threads();
		#pragma omp for schedule (static, chunk)
		for(i=0; i<Row; i++){
			for(j=0; j<Col; j++){
				a[i][j] = i+j;
				b[i][j] = i*j;
				c[i][j] = 0;
			}
		}
		/*Start multiplication */
		#pragma omp for schedule (static, chunk)
		for(i=0; i<Row; i++){
			for(j=0; j<Col; j++){
				tmp = 0;
				#pragma omp critical
				for(k=0; k<Col; k++)
				{
					c[i][j] += a[i][k]*b[j][k];
				}
			}
		}
	}
	//#pragma omp end parallel
	
	end = MPI_Wtime();
	printf("%f\n",end-start );
	//printf("%d\n",tid );
	//printf("%d\n",nthreads );
	return 0;
}

void print(float a[Row][Col],int row,int col){
	int i,j;
	for(i=0; i<row; i++){
			printf("\n");
			for(j=0; j<col; j++){
				printf("%f\t",a[i][j]);
			}
		}
}
