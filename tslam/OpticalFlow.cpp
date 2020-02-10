#include "Tslam.h"

int Tslam::initOpticalFlow() {
	flow = new Flow();
	flowScaling = 2.0f;
	int nLevel = 5;
	float fScale = 2.0f;
	int nWarpIters = 5;
	int nSolverIters = 20;
	float lambda = 1.0f;
	float theta = 0.33f;
	float tau = 0.125f;

	flowWidth = (int)(t265.width / flowScaling);
	flowHeight = (int)(t265.height / flowScaling);
	flow->initialize(flowWidth, flowHeight, lambda, theta, tau, nLevel, fScale, nWarpIters, nSolverIters);

	// Load fisheye Mask
	if (flowScaling == 2.0f) {
		fisheyeMask8 = cv::imread("maskHalf.png", cv::IMREAD_GRAYSCALE);
	}
	else {
		fisheyeMask8 = cv::imread("mask.png", cv::IMREAD_GRAYSCALE);
	}
	fisheyeMask8.convertTo(fisheyeMask, CV_32F, 1.0 / 255.0);
	flow->copyMaskToDevice(fisheyeMask);

	flowTemp = cv::Mat(flowHeight, flowWidth, CV_32FC2);
	flowTempRgb = cv::Mat(flowHeight, flowWidth, CV_32FC3);
	return 0;
}

int Tslam::solveOpticalFlow(cv::Mat im1, cv::Mat im2) {
	// Solve stereo depth
	cv::Mat equi1, equi2;
	cv::equalizeHist(im1, equi1);
	cv::equalizeHist(im2, equi2);
	//equi1 = im1;
	//equi2 = im2;

	if (flowScaling == 2.0f) {
		cv::Mat halfFisheye1, halfFisheye2;
		cv::resize(equi1, halfFisheye1, cv::Size(flowWidth, flowHeight));
		cv::resize(equi2, halfFisheye2, cv::Size(flowWidth, flowHeight));

		//cv::Mat depth = cv::Mat(stereoHeight, stereoWidth, CV_32F);
		clock_t start = clock();
		flow ->copyImagesToDevice(halfFisheye1, halfFisheye2);

		// Just TVL1
		flow->solveOpticalFlow();
		flow->copyFlowToHost(flowTemp);
		flow->copyFlowColorToHost(flowTempRgb, 20.0f);

		clock_t timeElapsed = (clock() - start);
		//std::cout << "time: " << timeElapsed << " ms" << std::endl;

		cv::Mat uvrgb8;
		flowTempRgb.convertTo(uvrgb8, CV_8UC3, 255.0f);
		//std::cout << uvrgb8.at<cv::Vec3b>(200, 200)[0] << std::endl;
		//std::cout << flowTemp.at<cv::Vec2f>(200, 200)[0] << std::endl;
		cv::putText(uvrgb8, std::to_string(timeElapsed), cv::Point(10, 12), cv::FONT_HERSHEY_PLAIN, 1, CV_RGB(0, 0, 0));
		cv::imshow("flow", uvrgb8);
	}
	else {
		//cv::Mat depth = cv::Mat(stereoHeight, stereoWidth, CV_32F);
	}
	return 0;
}