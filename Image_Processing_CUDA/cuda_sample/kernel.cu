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
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui_c.h>

using namespace cv;
using namespace std;

int main(int argc, char** argv){
	cv::Mat img = cv::imread("scar.jpg",CV_LOAD_IMAGE_UNCHANGED);

	if(img.empty()){
		cout<<"Error: Image not supported"<<endl;
		return -1;
	}

	else{

		int rows = img.rows;
		int cols = img.cols;
		cout<< cols <<" x "<<rows<<endl;
		//cv::namedWindow("Scarlett",CV_WINDOW_AUTOSIZE);
		//cv::imshow("Scarlett", img);
		cv::waitKey(0);
		cv::destroyAllWindows();
	}


}