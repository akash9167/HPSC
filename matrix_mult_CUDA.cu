/*
 *Author: Akash Bunde
 *HW3(HPSC)
 *
 *1.Using Naive Implementation
 *2.Using Shared Memory (TILED Approach)
 *
 *Implementing any program on cuda following steps to be checked
 *	-Set data initial data in Host
 *	-Copy data from Host to Device (cudamemcpy)
 *	-Allocate result memory in host (cudamalloc)
 *	-Set grid and block dimensions 
 *	-Execute kernel (<<< dimGrid, dimBlock >>>)
 *	-Copy result from Device to Host (cudamemcpy)
 */

#include <stdio.h>
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <math.h>
#include <sys\timeb.h>

#define TILE_WIDTH 50

static void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

/*matrix multiplication kernels*/

//non shared
__global__ void mat_mult( float *d_a , float *d_b , float *Pd , const int WIDTH ){
	
	unsigned int col = TILE_WIDTH*blockIdx.x + threadIdx.x ;
	unsigned int row = TILE_WIDTH*blockIdx.y + threadIdx.y ;
	
	for (int k = 0 ; k<WIDTH ; k++ ){
		Pd[row*WIDTH + col]+= d_a[row * WIDTH + k ] * d_b[ k * WIDTH + col] ;
    }
}

// shared
__global__ void shared_mat_mult( float *d_a , float *d_b , float *Pd , const int WIDTH ){
	
	__shared__ float d_as [TILE_WIDTH][TILE_WIDTH] ;
	__shared__ float d_bs [TILE_WIDTH][TILE_WIDTH] ;

	unsigned int col = TILE_WIDTH*blockIdx.x + threadIdx.x ;
	unsigned int row = TILE_WIDTH*blockIdx.y + threadIdx.y ;
	float Pval=0;
    
	for (int m = 0 ; m<WIDTH/TILE_WIDTH ; m++ ){
            
		d_as[threadIdx.y][threadIdx.x] =  d_a[row*WIDTH + (m*TILE_WIDTH + threadIdx.x)]  ;
		d_bs[threadIdx.y][threadIdx.x] =  d_b[ ( m*TILE_WIDTH + threadIdx.y) * WIDTH + col] ;
		
		__syncthreads() ; // wait till all threads have filled shared memory tile
           
		for ( int k = 0; k<TILE_WIDTH ; k++ )	Pval += d_as[threadIdx.y][k] * d_bs[k][threadIdx.x] ;
		
		__syncthreads() ; // synchronizing threads
		Pd[row*WIDTH + col] = Pval;
     }
}

int main (){
	
	float time;
	cudaEvent_t start, stop;

	HANDLE_ERROR( cudaEventCreate(&start) );
	HANDLE_ERROR( cudaEventCreate(&stop) );
	HANDLE_ERROR( cudaEventRecord(start, 0) );
	
	const int WIDTH = 10000;

	float h_a[WIDTH][WIDTH],
		  h_b[WIDTH][WIDTH],
		  h_result[WIDTH][WIDTH];

	float *d_a,
		  *d_b,
		  *d_result ; // device array
	int i , j ;
	//input in host array
	for ( i = 0 ; i<WIDTH ; i++ ){
		for (j = 0 ; j<WIDTH ; j++){
			h_a[i][j] = i*j ;
			h_b[i][j] = i+j ;
		}
	}

	cudaMalloc((void **) &d_a , WIDTH*WIDTH*sizeof (int) ) ;
	cudaMalloc((void **) &d_b , WIDTH*WIDTH*sizeof (int) ) ;

	cudaMemcpy ( d_a , h_a , WIDTH*WIDTH*sizeof (int) , cudaMemcpyHostToDevice ) ;
	cudaMemcpy ( d_b , h_b , WIDTH*WIDTH*sizeof (int) , cudaMemcpyHostToDevice ) ;

	cudaMalloc((void **) &d_result , WIDTH*WIDTH*sizeof (int) ) ;
	
	dim3 dimGrid ( WIDTH/TILE_WIDTH , WIDTH/TILE_WIDTH ,1 ) ;
	dim3 dimBlock( TILE_WIDTH, TILE_WIDTH, 1 ) ;

// Change if 0 to if 1 for running non shared code and make if 0 for shared memory code
#if 0
	mat_mult <<<dimGrid,dimBlock>>> ( d_a , d_b ,d_result , WIDTH) ;
#endif
 
#if 1
	shared_mat_mult<<<dimGrid,dimBlock>>> ( d_a , d_b ,d_result , WIDTH) ;
#endif

	cudaMemcpy(h_result , d_result , WIDTH*WIDTH*sizeof(int) , cudaMemcpyDeviceToHost) ;
/*
	for ( i = 0 ; i<WIDTH ; i++ ){
		for ( j = 0 ; j < WIDTH ; j++ ){
			printf ("%0.1f  ",h_result[i][j] ) ;
		}
		printf ("\n") ;
	}
*/

	HANDLE_ERROR( cudaEventRecord(stop, 0) );
	HANDLE_ERROR( cudaEventSynchronize(stop) );
	HANDLE_ERROR( cudaEventElapsedTime(&time, start, stop) );
	printf("Time to generate:  %f ms \n", time);
	
	cudaFree(h_a);
	cudaFree(h_b);
	cudaFree(h_result);
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_result);

	return 0;
}
