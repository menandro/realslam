#include <opencv2/opencv.hpp>
#include <opticalflow/flow.h>

#include <time.h>

#if _WIN64
#define LIB_PATH "D:/dev/lib64/"
#define CV_LIB_PATH "D:/dev/lib64/"
#else
#define LIB_PATH "D:/dev/staticlib32/"
#endif

#ifdef _DEBUG
#define LIB_EXT "d.lib"
#else
#define LIB_EXT ".lib"
#endif

#define CUDA_LIB_PATH "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/lib/x64/"
#pragma comment(lib, CUDA_LIB_PATH "cudart.lib")

#define CV_VER_NUM CVAUX_STR(CV_MAJOR_VERSION) CVAUX_STR(CV_MINOR_VERSION) CVAUX_STR(CV_SUBMINOR_VERSION)
#pragma comment(lib, LIB_PATH "opencv_core" CV_VER_NUM LIB_EXT)
#pragma comment(lib, LIB_PATH "opencv_highgui" CV_VER_NUM LIB_EXT)
#pragma comment(lib, LIB_PATH "opencv_videoio" CV_VER_NUM LIB_EXT)
#pragma comment(lib, LIB_PATH "opencv_imgproc" CV_VER_NUM LIB_EXT)
#pragma comment(lib, LIB_PATH "opencv_calib3d" CV_VER_NUM LIB_EXT)
#pragma comment(lib, LIB_PATH "opencv_xfeatures2d" CV_VER_NUM LIB_EXT)
#pragma comment(lib, LIB_PATH "opencv_optflow" CV_VER_NUM LIB_EXT)
#pragma comment(lib, LIB_PATH "opencv_imgcodecs" CV_VER_NUM LIB_EXT)
#pragma comment(lib, LIB_PATH "opencv_features2d" CV_VER_NUM LIB_EXT)
#pragma comment(lib, LIB_PATH "opencv_tracking" CV_VER_NUM LIB_EXT)
#pragma comment(lib, LIB_PATH "opencv_flann" CV_VER_NUM LIB_EXT)
#pragma comment(lib, LIB_PATH "opencv_cudafeatures2d" CV_VER_NUM LIB_EXT)
#pragma comment(lib, LIB_PATH "opencv_cudaimgproc" CV_VER_NUM LIB_EXT)
#pragma comment(lib, LIB_PATH "opencv_video" CV_VER_NUM LIB_EXT)

int test_sequence() {
	std::string mainfolder = "h:/data_rs_iis/20190909";
	Flow* flow = new Flow();
	float flowScaling = 1.0f;
	int width = 848;
	int height = 800;
	int nLevel = 12;
	float fScale = 1.2f;
	int nWarpIters = 100;
	int nSolverIters = 100;
	float lambda = 1.0f;
	float theta = 0.33f;
	float tau = 0.125f;

	int flowWidth = (int)(width / flowScaling);
	int flowHeight = (int)(height / flowScaling);
	flow->initialize(flowWidth, flowHeight, lambda, theta, tau, nLevel, fScale, nWarpIters, nSolverIters);

	// Load fisheye Mask
	cv::Mat fisheyeMask8, fisheyeMask;
	if (flowScaling == 2.0f) {
		fisheyeMask8 = cv::imread("maskHalf.png", cv::IMREAD_GRAYSCALE);
	}
	else {
		fisheyeMask8 = cv::imread("mask.png", cv::IMREAD_GRAYSCALE);
	}
	fisheyeMask8.convertTo(fisheyeMask, CV_32F, 1.0 / 255.0);
	flow->copyMaskToDevice(fisheyeMask);

	cv::Mat flowTemp = cv::Mat(flowHeight, flowWidth, CV_32FC2);
	cv::Mat flowTempRgb = cv::Mat(flowHeight, flowWidth, CV_32FC3);

	for (int k = 1; k <= 771; k++) {
		cv::Mat im1 = cv::imread(mainfolder + "/colored_0/data/im" + std::to_string(k) + ".png", cv::IMREAD_GRAYSCALE);
		cv::Mat im2 = cv::imread(mainfolder + "/colored_0/data/im" + std::to_string(k-1) + ".png", cv::IMREAD_GRAYSCALE);
		
		cv::Mat equi1, equi2;
		//cv::equalizeHist(im1, equi1);
		//cv::equalizeHist(im2, equi2);

		cv::Mat halfFisheye1, halfFisheye2;
		cv::resize(im1, halfFisheye1, cv::Size(flowWidth, flowHeight));
		cv::resize(im2, halfFisheye2, cv::Size(flowWidth, flowHeight));

		//cv::Mat depth = cv::Mat(stereoHeight, stereoWidth, CV_32F);
		clock_t start = clock();
		flow->copyImagesToDevice(halfFisheye1, halfFisheye2);

		// Just TVL1
		flow->solveOpticalFlow();
		flow->copyFlowToHost(flowTemp);
		flow->copyFlowColorToHost(flowTempRgb, 20.0f);

		clock_t timeElapsed = (clock() - start);

		std::string appender;
		if (k < 10) appender = "000";
		else if ((k >= 10) && (k < 100)) appender = "00";
		else if ((k >= 100) && (k < 1000)) appender = "0";
		else appender = "";
		cv::writeOpticalFlow(mainfolder + "/flow_0/im" + appender + std::to_string(k) + ".flo", flowTemp);

		cv::Mat uvrgb8;
		flowTempRgb.convertTo(uvrgb8, CV_8UC3, 255.0f);
		cv::imwrite(mainfolder + "/flow_0/vis/im" + appender + std::to_string(k) + ".png", uvrgb8);
		
		cv::putText(uvrgb8, std::to_string(timeElapsed), cv::Point(10, 12), cv::FONT_HERSHEY_PLAIN, 1, CV_RGB(0, 0, 0));
		cv::imshow("flow", uvrgb8);
		cv::waitKey(1);
	}
	cv::waitKey();
	return 0;
}


int main() {
	return test_sequence();
}