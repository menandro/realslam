#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <vector>
#include <stdio.h>
#include <stdlib.h>

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
#pragma comment(lib, CV_LIB_PATH "opencv_core" CV_VER_NUM LIB_EXT)
#pragma comment(lib, CV_LIB_PATH "opencv_highgui" CV_VER_NUM LIB_EXT)
#pragma comment(lib, CV_LIB_PATH "opencv_videoio" CV_VER_NUM LIB_EXT)
#pragma comment(lib, LIB_PATH "opencv_imgproc" CV_VER_NUM LIB_EXT)
#pragma comment(lib, LIB_PATH "opencv_calib3d" CV_VER_NUM LIB_EXT)
#pragma comment(lib, LIB_PATH "opencv_xfeatures2d" CV_VER_NUM LIB_EXT)
#pragma comment(lib, LIB_PATH "opencv_optflow" CV_VER_NUM LIB_EXT)
#pragma comment(lib, LIB_PATH "opencv_imgcodecs" CV_VER_NUM LIB_EXT)
#pragma comment(lib, LIB_PATH "opencv_features2d" CV_VER_NUM LIB_EXT)
#pragma comment(lib, LIB_PATH "opencv_tracking" CV_VER_NUM LIB_EXT)

int main() {
	cv::Mat depthImage;
	cv::Mat binaryImage;
	cv::RNG rng;
	rng = cv::RNG(12345);

	cv::Mat im = cv::imread("h:/data_rs_iis/20190909/colored_0/data/im303.png", cv::IMREAD_GRAYSCALE);
	depthImage = im;
	cv::imshow("im", im);
	
	cv::threshold(depthImage, binaryImage, 128, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
	cv::imshow("binary", binaryImage);
	

	std::vector<std::vector<cv::Point> > contours;
	std::vector<cv::Vec4i> hierarchy;
	cv::findContours(depthImage, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0));

	// Detect largest size contour
	int maxCont = 0;
	int indMax = 0;
	for (int i = 0; i < contours.size(); i++) {
		std::cout << contours[i].size() << std::endl;
		if (contours[i].size() > maxCont) {
			indMax = i;
			maxCont = contours[i].size();
		}
	}
	std::cout << "max" << contours[indMax].size() << std::endl;
	/// Draw contours
	cv::Mat drawing = cv::Mat::zeros(binaryImage.size(), CV_8UC3);
	
	for (int i = 0; i < contours.size(); i++) {
		cv::Scalar color = cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
		drawContours(drawing, contours, i, color, 2, 8, hierarchy, 0, cv::Point());
	}
	//for (int i = 0; i< contours.size(); i++)
	//{
	//	cv::Scalar color = cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
	//	drawContours(drawing, contours, i, color, 2, 8, hierarchy, 0, cv::Point());
	//	//std::cout << contours[i] << std::endl;
	//}

	/// Show in a window
	cv::namedWindow("Contours", cv::WINDOW_AUTOSIZE);
	imshow("Contours", drawing);
	cv::waitKey();
	return 0;
}