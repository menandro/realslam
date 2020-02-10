#include <opencv2/opencv.hpp>
#include <time.h>
#include "../directalignment/directalignment.h"
#include "../stereotgv/stereotgv.h"
#include "../stereolite/stereolite.h"

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

#define CUDA_LIB_PATH "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.1/lib/x64/"
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

void showDepthJetExponential(std::string windowName, cv::Mat image, float maxDepth, float curve, bool shouldWait = true) {
	cv::Mat u_norm, u_exp, u_gray, u_color;
	//std::cout << image.at<float>(400, 400) << std::endl;
	u_norm = image / maxDepth;
	//std::cout << u_norm.at<float>(400, 400) << std::endl;
	//float curver = 0.1f;
	u_exp = 1.0f - curve / (curve + u_norm);
	u_exp = u_exp * 255.0f;
	u_exp.convertTo(u_gray, CV_8UC1);
	cv::applyColorMap(u_gray, u_color, cv::COLORMAP_JET);

	cv::imshow(windowName, u_color);
	if (shouldWait) cv::waitKey();
}

int test_FlowAndStereoLite(){
	std::string mainfolder = "h:/data_rs_iis/20190913_2";
	StereoLite * stereolite = new StereoLite();
	int width = 848;
	int height = 800;
	int nLevel = 4;
	float fScale = 2.0f;
	int nWarpIters = 100;
	int nSolverIters = 100;
	float lambda = 1.0f;
	float theta = 0.33f;
	float tau = 0.125f;
	stereolite->limitRange = 0.2f;
	float maxDepthVis = 20.0f;

	int stereoWidth = width;
	int stereoHeight = height;
	stereolite->baseline = 0.0642f;
	stereolite->focal = 285.8557f;
	cv::Mat translationVector, calibrationVector;
	translationVector = cv::readOpticalFlow("../test_rstracking/translationVector.flo");
	calibrationVector = cv::readOpticalFlow("../test_rstracking/calibrationVector.flo");

	stereolite->initialize(stereoWidth, stereoHeight, lambda, theta, tau, nLevel, fScale, nWarpIters, nSolverIters);

	cv::Mat mask = cv::Mat::zeros(cv::Size(stereoWidth, stereoHeight), CV_8UC1);
	circle(mask, cv::Point(stereoWidth / 2, stereoHeight / 2), stereoWidth / 2 - (int)(40), cv::Scalar(256.0f), -1);
	cv::Mat fisheyeMask;
	mask.convertTo(fisheyeMask, CV_32F, 1.0 / 255.0);

	stereolite->copyMaskToDevice(fisheyeMask);

	// Load vector fields
	stereolite->loadVectorFields(translationVector, calibrationVector);

	cv::Mat equi1, equi2;
	cv::Mat depth = cv::Mat(stereoHeight, stereoWidth, CV_32F);
	cv::Mat depthVis;

	clock_t avetime = 0;

	for (int k = 150; k <= 150; k++) {
		cv::Mat im1 = cv::imread(mainfolder + "/colored_0/data/im" + std::to_string(k) + ".png", cv::IMREAD_GRAYSCALE);
		cv::Mat im2 = cv::imread(mainfolder + "/colored_1/data/im" + std::to_string(k) + ".png", cv::IMREAD_GRAYSCALE);

		// Solve stereo depth
		cv::equalizeHist(im1, equi1);
		cv::equalizeHist(im2, equi2);

		clock_t start = clock();
		stereolite->copyImagesToDevice(equi1, equi2);
		stereolite->solveStereoForwardMasked();
		stereolite->copyStereoToHost(depth);
		clock_t timeElapsed = (clock() - start);
		std::cout << "time: " << timeElapsed << " ms ";

		cv::Mat uvrgb = cv::Mat(im1.size(), CV_32FC3);
		stereolite->copyDisparityVisToHost(uvrgb, 5.0f);
		cv::imshow("flow", uvrgb);

		depth.copyTo(depthVis, mask);
		showDepthJetExponential("color", depthVis, maxDepthVis, 0.1f, false);
		//showDepthJet("color", depthVis, 3.0f, false);

		std::string appender;
		if (k < 10) appender = "000";
		else if ((k >= 10) && (k < 100)) appender = "00";
		else if ((k >= 100) && (k < 1000)) appender = "0";
		else appender = "";
		//saveDepthJet(mainfolder + outputfolder + "/im" + appender + std::to_string(k) + ".png", depthVis, 5.0f);

		cv::waitKey(1);
	}
	cv::waitKey(0);
	return 0;
}

int test_FlowAndStereoTgv_FAIL() {
	StereoTgv * stereotgv = new StereoTgv();
	int width = 848;
	int height = 800;
	int nLevel = 5;
	float fScale = 2.0f;// 2.0f;
	int nWarpIters = 20;// 10;
	int nSolverIters = 10;// 20;
	float lambda = 5.0f;
	stereotgv->limitRange = 0.1f;
	float maxDepthVis = 20.0f;

	int stereoWidth = width;
	int stereoHeight = height;
	stereotgv->baseline = 0.0642f;
	stereotgv->focal = 285.8557f;
	cv::Mat translationVector, calibrationVector;
	std::string mainfolder = "h:/data_rs_iis/";
	std::string dataset = "20190913_2";
	translationVector = cv::readOpticalFlow(mainfolder + "translationVector.flo");
	calibrationVector = cv::readOpticalFlow(mainfolder + "calibrationVector.flo");

	float beta = 4.0f;
	float gamma = 0.2f;
	float alpha0 = 17.0f;
	float alpha1 = 1.2f;
	float timeStepLambda = 1.0f;

	stereotgv->initialize(stereoWidth, stereoHeight, beta, gamma, alpha0, alpha1,
		timeStepLambda, lambda, nLevel, fScale, nWarpIters, nSolverIters);
	stereotgv->visualizeResults = true;

	cv::Mat mask = cv::Mat::zeros(cv::Size(stereoWidth, stereoHeight), CV_8UC1);
	circle(mask, cv::Point(stereoWidth / 2, stereoHeight / 2), stereoWidth / 2 - (int)(40), cv::Scalar(256.0f), -1);
	cv::Mat fisheyeMask;
	mask.convertTo(fisheyeMask, CV_32F, 1.0 / 255.0);

	stereotgv->copyMaskToDevice(fisheyeMask);

	// Load vector fields
	stereotgv->loadVectorFields(translationVector, calibrationVector);

	cv::Mat equi1, equi2;

	for (int k = 150; k <= 150; k++) {
		cv::Mat im1 = cv::imread(mainfolder + dataset + "/colored_0/data/im" + std::to_string(k) + ".png", cv::IMREAD_GRAYSCALE);
		cv::Mat im2 = cv::imread(mainfolder + dataset + "/colored_1/data/im" + std::to_string(k) + ".png", cv::IMREAD_GRAYSCALE);

		// Solve stereo depth
		cv::equalizeHist(im1, equi1);
		cv::equalizeHist(im2, equi2);
		cv::GaussianBlur(equi1, equi1, cv::Size(5, 5), 0.1, 0.0);
		cv::GaussianBlur(equi2, equi2, cv::Size(5, 5), 0.1, 0.0);

		clock_t start = clock();
		stereotgv->copyImagesToDevice(equi1, equi2);
		stereotgv->solveStereoForwardMasked();

		cv::Mat depth = cv::Mat(stereoHeight, stereoWidth, CV_32F);
		cv::Mat depthVis;
		stereotgv->copyStereoToHost(depth);

		clock_t timeElapsed = (clock() - start);
		std::cout << "time: " << timeElapsed << " ms" << std::endl;

		//stereotgv->copy1DDisparityToHost(depth);
		depth.copyTo(depthVis, mask);
		showDepthJetExponential("color", depthVis, maxDepthVis, 0.1f, false);

		std::string appender;
		if (k < 10) appender = "000";
		else if ((k >= 10) && (k < 100)) appender = "00";
		else if ((k >= 100) && (k < 1000)) appender = "0";
		else appender = "";

		cv::Mat depth16;
		depthVis.convertTo(depth16, CV_16U, 256.0f);
		cv::imwrite(mainfolder + "/directdepth_0/im" + appender + std::to_string(k) + ".png", depth16);
		//saveDepthJetExponential(mainfolder + "/error/ours/im" + appender + std::to_string(k) + ".png", depthVis, maxDepthVis, 0.1f);
		std::cout << k << std::endl;
		cv::waitKey(1);
	}
	cv::waitKey();
	return 0;
}

int test_DirectDenseFlow() {
	DirectAlignment *da = new DirectAlignment();
	da->initialize(848, 800, 1.0f, 0.33f, 0.125f, 400);
	da->gradThreshold = 0.10f;
	std::string mainfolder = "H:/data_rs_iis/20190913_2/";
	for (int k = 150; k < 151; k++) {
		std::string filename1 = "im" + std::to_string(k);
		std::string filename2 = "im" + std::to_string(k - 1);
		std::cout << filename1 << " " << filename2 << std::endl;

		cv::Mat input1 = cv::imread(mainfolder + "colored_0/data/" + filename1 + ".png");
		cv::Mat input2 = cv::imread(mainfolder + "colored_0/data/" + filename2 + ".png");
		std::string outputflow = mainfolder + "directstereo_0/" + filename1 + ".flo";
		std::string outputflowvis = mainfolder + "error/directstereo_0/" + filename1 + ".png";

		cv::cvtColor(input1, input1, cv::COLOR_RGB2GRAY);
		cv::cvtColor(input2, input2, cv::COLOR_RGB2GRAY);

		cv::resize(input1, input1, cv::Size(848, 800));
		cv::resize(input2, input2, cv::Size(848, 800));
		cv::equalizeHist(input1, input1);
		cv::equalizeHist(input2, input2);
		cv::Mat mask = cv::Mat::zeros(input1.size(), CV_32FC1) + 1.0f;

		clock_t start = clock();
		da->copyImagesToDevice(input1, input2);
		da->copyMaskToDevice(mask);
		da->solveDirectFlow();

		cv::Mat flow = cv::Mat(input1.size(), CV_32FC2);
		da->copyFlowToHost(flow);
		clock_t timeElapsed = (clock() - start);
		std::cout << "time: " << timeElapsed << " ms" << std::endl;
		cv::writeOpticalFlow(outputflow, flow);

		cv::Mat uvrgb = cv::Mat(input1.size(), CV_32FC3);
		da->copyFlowColorToHost(uvrgb, 5.0f);

		cv::Mat uvrgb8;
		uvrgb.convertTo(uvrgb8, CV_8UC3, 255.0f);
		cv::imshow("test", uvrgb8);
		cv::imwrite(outputflowvis, uvrgb8);
		cv::waitKey(1);
	}
	cv::waitKey();
	return 0;
}

int test_solveDepthEquirectangular() {
	DirectAlignment *da = new DirectAlignment();
	da->initialize(640, 360, 1.0f, 0.33f, 0.125f, 100);
	da->gradThreshold = 0.10f;
	std::string mainfolder = "h:/data_chiba/";

	for (int k = 1; k < 4228; k++) {
		std::string filename1 = std::string(4 - std::to_string(k).length(), '0') + std::to_string(k);
		std::string filename2 = std::string(4 - std::to_string(k - 1).length(), '0') + std::to_string(k - 1);
		std::cout << filename1 << " " << filename2 << std::endl;

		cv::Mat input1 = cv::imread(mainfolder + "colored_0/data/" + filename1 + ".png");
		cv::Mat input2 = cv::imread(mainfolder + "colored_0/data/" + filename2 + ".png");
		std::string outputflow = mainfolder + "direct_0/" + filename1 + ".flo";
		std::string outputflowvis = mainfolder + "error/direct_0/" + filename1 + ".png";

		cv::cvtColor(input1, input1, cv::COLOR_RGB2GRAY);
		cv::cvtColor(input2, input2, cv::COLOR_RGB2GRAY);

		cv::resize(input1, input1, cv::Size(640, 360));
		cv::resize(input2, input2, cv::Size(640, 360));
		cv::equalizeHist(input1, input1);
		cv::equalizeHist(input2, input2);
		cv::Mat mask = cv::Mat::zeros(input1.size(), CV_32FC1) + 1.0f;

		da->copyImagesToDevice(input1, input2);
		da->copyMaskToDevice(mask);
		da->solveDirectFlow();

		cv::Mat flow = cv::Mat(input1.size(), CV_32FC2);
		da->copyFlowToHost(flow);
		cv::writeOpticalFlow(outputflow, flow);

		cv::Mat uvrgb = cv::Mat(input1.size(), CV_32FC3);
		da->copyFlowColorToHost(uvrgb, 1.0f);

		cv::Mat uvrgb8;
		uvrgb.convertTo(uvrgb8, CV_8UC3, 255.0f);
		cv::imshow("test", uvrgb8);
		cv::imwrite(outputflowvis, uvrgb8);
		cv::waitKey(1);
	}
	cv::waitKey();
	return 0;
}

int main() {
	//test_FlowAndStereoLite();
	test_DirectDenseFlow();
	//test_solveDepthEquirectangular();
}

//int main() {
//	DirectAlignment *da = new DirectAlignment();
//	da->initialize(640, 360, 1.0f, 0.33f, 0.125f, 100);
//	da->gradThreshold = 0.10f;
//	std::string mainfolder = "h:/data_chiba/";
//
//	for (int k = 1; k < 4228; k++) {
//		std::string filename1 = std::string(4 - std::to_string(k).length(), '0') + std::to_string(k);
//		std::string filename2 = std::string(4 - std::to_string(k-1).length(), '0') + std::to_string(k-1);
//		std::cout << filename1 << " " << filename2 << std::endl;
//
//		cv::Mat input1 = cv::imread(mainfolder + "colored_0/data/" + filename1 + ".png");
//		cv::Mat input2 = cv::imread(mainfolder + "colored_0/data/" + filename2 + ".png");
//		std::string outputflow = mainfolder + "direct_0/" + filename1 + ".flo";
//		std::string outputflowvis = mainfolder + "error/direct_0/" + filename1 + ".png";
//
//		cv::cvtColor(input1, input1, cv::COLOR_RGB2GRAY);
//		cv::cvtColor(input2, input2, cv::COLOR_RGB2GRAY);
//
//		cv::resize(input1, input1, cv::Size(640, 360));
//		cv::resize(input2, input2, cv::Size(640, 360));
//		cv::equalizeHist(input1, input1);
//		cv::equalizeHist(input2, input2);
//		cv::Mat mask = cv::Mat::zeros(input1.size(), CV_32FC1) + 1.0f;
//
//		da->copyImagesToDevice(input1, input2);
//		da->copyMaskToDevice(mask);
//		da->solveDirectFlow();
//
//		cv::Mat flow = cv::Mat(input1.size(), CV_32FC2);
//		da->copyFlowToHost(flow);
//		cv::writeOpticalFlow(outputflow, flow);
//
//		cv::Mat uvrgb = cv::Mat(input1.size(), CV_32FC3);
//		da->copyFlowColorToHost(uvrgb, 1.0f);
//
//		cv::Mat uvrgb8;
//		uvrgb.convertTo(uvrgb8, CV_8UC3, 255.0f);
//		cv::imshow("test", uvrgb8);
//		cv::imwrite(outputflowvis, uvrgb8);
//		cv::waitKey(1);
//	}
//	cv::waitKey();
//}