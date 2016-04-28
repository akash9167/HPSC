#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_image.h>
#include <helper_timer.h>
#include <helper_cuda.h>
#include <opencv2\opencv.hpp>
#include <opencv2\highgui\highgui.hpp>

using namespace cv;
using namespace std;
/*
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

__global__ void filter(uchar *d_data){
	int t = blockIdx.x*blockDim.x + threadIdx.x;
	d_data[t] = 100;
}
*/
int main(int argc, char** argv){
	const Mat img = cv::imread("scar1.jpg",CV_LOAD_IMAGE_UNCHANGED);
	Mat img2 = img.clone();
	Mat img3 = img.clone();

	if(img.empty()){
		cout<<"Error: Image not supported"<<endl;
		return -1;
	}

	else{
		uchar *input = img.data;

		int rows = img.rows,
			cols = img.cols,
			step = img.step,
			rows_d, cols_d, step_d;

		int channels = img.channels();
		std::cout<< cols <<" x "<<rows<<"Step Size: "<<step<<endl;
		if(img.depth()==CV_8U) cout << "Unsigned char image" << endl;
		std::cout<<"Number of channels: "<<img.channels()<<endl;
		std::cout<<"Is data continuous: "<<img.isContinuous()<<endl;
		size_t total_size = rows*cols*3*sizeof(uchar);
		
		uchar *input_d, *output_d;
		
		const unsigned int kr = 8;
		const int ks = 2*kr+1;
		double sigs=20;
		double sigr=20;
		double g[ks][ks],d[ks][ks],gdist[ks][ks];
		uchar I[ks][ks], a[ks][ks];
		uchar *input2 = img2.data;
		
		double norm_gdist=0;

		for(int i=0; i<ks; i++){
			for(int j=0; j<ks; j++){
				d[i][j] = (kr-i)*(kr-i)+(kr-j)*(kr-j);
				float den = 1/(sigs*sqrt(2*3.142));
				float exp_=-d[i][j]/(2*sigs*sigs);
				gdist[i][j]=den*exp(exp_);
				norm_gdist += gdist[i][j];
				//cout<<d[i][j]<<"   ";
			}
			//cout<<endl;
		}
/*
		cudaMemcpy(input_d, input, total_size,cudaMemcpyHostToDevice);
		cudaMemcpy(&rows_d, &rows, sizeof(int),cudaMemcpyHostToDevice);
		cudaMemcpy(&cols_d, &cols, sizeof(int),cudaMemcpyHostToDevice);
		cudaMemcpy(&step_d, &step, sizeof(int),cudaMemcpyHostToDevice);
		
		cudaMalloc((void**)input_d,	total_size);
		cudaMalloc((void**)rows_d,	sizeof(int));
		cudaMalloc((void**)cols_d,	sizeof(int));
		cudaMalloc((void**)step_d,	sizeof(int));
		cudaMalloc((void**)output_d,total_size);
		cudaMalloc((void**)output_d,total_size);
*/
/*
		uchar * d_data;
		unsigned int size = channels*rows*cols*sizeof(uchar);
		
		cudaMalloc((void**)&d_data, size);
		cudaMemcpy( d_data, img.data, size, cudaMemcpyHostToDevice);
	
		float time;
		cudaEvent_t start, stop;

		HANDLE_ERROR( cudaEventCreate(&start) );
		HANDLE_ERROR( cudaEventCreate(&stop) );
		HANDLE_ERROR( cudaEventRecord(start, 0) );

		//filter<<<rows, channels*cols>>>(d_data);
		
		cudaThreadSynchronize();
		
		HANDLE_ERROR( cudaEventRecord(stop, 0) );
		HANDLE_ERROR( cudaEventSynchronize(stop) );
		HANDLE_ERROR( cudaEventElapsedTime(&time, start, stop) );
		printf("Time to generate:  %f ms \n", time);
		//cudaMemcpy(img.data, d_data, size, cudaMemcpyDeviceToHost);
		
*/
			
			int l=rows*step;
/*		
			for(int y=0; y<rows; y++)
				for(int x=0; x<cols; x++){
					img_2D[y][x]=input[y*step+channels*x];
					img_2D[y][x+1]=input[y*step+channels*x+1];
					img_2D[y][x+2]=input[y*step+channels*x+2];
				}
*/				

		for(int i=0; i<l; i++){
			int x=(i%step);
			int y=(i/step);
			for(int p=0; p<kr; p++){
				for(int q=0; q<kr; q++){
					if(x-channels*q<0 || y-p<0 || x+channels*q>=l || y+p>=l || i+p*step+channels*q>=l ||i+p*step-channels*q>=l||i-p*step+channels*q>=l){
						I[kr+p][kr+q]=input[i];
						a[kr+p][kr+q]=0;

						I[kr-p][kr+q]=input[i];
						a[kr-p][kr+q]=0;

						I[kr+p][kr-q]=input[i];
						a[kr+p][kr-q]=0;

						I[kr-p][kr-q]=input[i];
						a[kr-p][kr-q]=0;
						}

						else{
							a[kr+p][kr+q]=input[i+p*step+channels*q];
							I[kr+p][kr+q]=a[kr+p][kr+q]-a[kr][kr];

							a[kr-p][kr+q]=input[i-p*step+channels*q];
							I[kr-p][kr+q]=a[kr-p][kr+q]-a[kr][kr];

							a[kr+p][kr-q]=input[i+p*step-channels*q];
							I[kr+p][kr-q]=a[kr+p][kr-q]-a[kr][kr];

							a[kr-p][kr-q]=input[i-p*step-channels*q];
							I[kr-p][kr-q]=a[kr-p][kr-q]-a[kr][kr];
							
						}
			//			std::cout<<(int)I[p][q]<<"  ";
				}
			//	std::cout<<endl;
			}
					/*					else{
						a[kr-p][kr-q]= input[i-p*step-channels*q];
						I[kr-p][kr-q]= input[i-p*step-channels*q]-input[i];
					//	cout<<"I[][]: "<<(int)I[kr+p][kr+q]<<endl;
					}

					if(i-channels*q<0){
						I[kr][kr+q]=0;
						a[kr][kr+q]=0;
//						cout<<"I[][]: "<<(int)I[kr+p][kr+q]<<endl;
					}
					else{
						a[kr-p][kr+q]=input[i-p*step+channels*q];
						I[kr-p][kr+q]=input[i-p*step+channels*q]-input[i];
//						cout<<"I[][]: "<<(int)I[kr+p][kr+q]<<endl;
					}

					if((i+p*step-channels*q<0)||(i+p*step-channels*q>=step)){
						I[kr+p][kr-q]=0;
						a[kr+p][kr-q]=0;
//						cout<<"I[][]: "<<(int)I[kr+p][kr+q]<<endl;
					}
					else{
						a[kr+p][kr-q]=input[i+p*step-channels*q];
						I[kr+p][kr-q]=input[i+p*step-channels*q]-input[i];
//						cout<<"I[][]: "<<(int)I[kr+p][kr+q]<<endl;
					}

					if((i+p*step+channels*q<0)||(i+p*step+channels*q>=step)){
						I[kr+p][kr+q]=0;
						a[kr+p][kr+q]=0;
						//cout<<"I[][]: "<<(int)I[kr+p][kr+q]<<endl;
					}
					else{
						a[kr+p][kr+q]=input[i+p*step+channels*q];
						I[kr+p][kr+q]=input[i+p*step+channels*q]-input[i];
					}
					
				}
			}
			
			//k=i%step;
			//j=(k-i*step)%channels;
*/			
			
		
			double sum=0;
			for(int t=0; t<ks;t++)
				for(int j=0;j<ks;j++){
				//cout<<endl<<g[t][j]<<"initial"<<endl;
				g[t][j]=(1/(sigr*sqrt(2*3.142)))*exp(-((I[t][j]*I[t][j]))/(2*sigr*sigr));
				
				//cout<<endl<<g[t][j]<<"final"<<endl;				
			}
				//cout<<endl<<sum<<"final"<<endl;	

			double answer=0;
			for(int q=0;q<ks;q++)
				for(int w=0;w<ks;w++){
					answer+=a[q][w]*gdist[q][w]*g[q][w];
					sum+=g[q][w]*gdist[q][w];
					//cout<<answer/sum<<"ansmwer"<<endl;
				}

			input2[i]= (uchar)(answer/sum);
		}

		///////////////////////////////////////////////////////
/*		for (int y=0; y<rows; y++){
			for(int x=0; x<channels*cols; x++){

				for(int p=0; p<kr; p++){
					for(int q=0; q<kr; q++){
						if(x-channels*q<0 || y-p<0 || x+channels*q>=l || y+p>=l){
							I[p][q]=img_2D[y][x];
							a[p][q]=0;
						}

						else{
							a[kr+p][kr+q]=img_2D[y+p][x+channels*q];
							I[kr+p][kr+q]=a[kr+p][kr+q]-a[kr][kr];

							a[kr-p][kr+q]=img_2D[y-p][x+channels*q];
							I[kr-p][kr+q]=a[kr-p][kr+q]-a[kr][kr];

							a[kr+p][kr-q]=img_2D[y+p][x-channels*q];
							I[kr+p][kr-q]=a[kr+p][kr-q]-a[kr][kr];

							a[kr-p][kr-q]=img_2D[y-p][x-channels*q];
							I[kr-p][kr-q]=a[kr-p][kr-q]-a[kr][kr];
							
						}
					}
						double norm_g=0, answer=0;
						for(int i=0; i<ks;i++){
							for(int j=0;j<ks;j++){
								g[i][j]=gdist[i][j]*exp(-I[i][j]*I[i][j]/(2*sigr*sigr));
								answer += a[i][j]*g[i][j];
								norm_g += g[i][j];
								//cout<<gdist[i][j]<<"   ";
							}
							//cout<<endl;
						}
						uchar total = (uchar)(answer/norm_g);
						img_2D[y][x] = total;
					}
				}
		}

		for(int y=0; y<rows; y++)
				for(int x=0; x<cols; x++){
					input[y*step+channels*x]=img_2D[y][x];
					input[y*step+channels*x+1]=img_2D[y][x+1];
					input[y*step+channels*x+2]=img_2D[y][x+2];
				}
*/
		cv::namedWindow("Original",CV_WINDOW_AUTOSIZE);
		cv::imshow("Original", img);
		cv::namedWindow("Sampled",CV_WINDOW_AUTOSIZE);
		cv::imshow("Sampled", img2);
		cv::waitKey(0);
		cv::destroyAllWindows();		

	}
}
