#include "fisheyeslam.h"

int FisheyeSlam::initOpticalFlow() {
	flow = new Flow();
	int nLevel = 5;
	float fScale = 2.0f;
	int nWarpIters = 5;
	int nSolverIters = 20;
	float lambda = 1.0f;
	float theta = 0.33f;
	float tau = 0.125f;

	flowWidth = this->width;
	flowHeight = this->height;
	flow->initialize(flowWidth, flowHeight, lambda, theta, tau, nLevel, fScale, nWarpIters, nSolverIters);
	
	flow->copyMaskToDevice(fisheyeMask32fc1);

	flowTemp = cv::Mat(flowHeight, flowWidth, CV_32FC2);
	flowTempMasked = cv::Mat(flowHeight, flowWidth, CV_32FC2);
	flowTempRgb = cv::Mat(flowHeight, flowWidth, CV_32FC3);

	prevImage = cv::Mat::zeros(flowHeight, flowWidth, CV_8UC1);
	return 0;
}

int FisheyeSlam::solveOpticalFlow(cv::Mat im1, cv::Mat im2) {
	// Solve stereo depth
	cv::Mat equi1, equi2;
	cv::equalizeHist(im1, equi1);
	cv::equalizeHist(im2, equi2);

	//cv::Mat depth = cv::Mat(stereoHeight, stereoWidth, CV_32F);
	clock_t start = clock();
	flow->copyImagesToDevice(equi1, equi2);

	// Just TVL1
	flow->solveOpticalFlow();
	flow->copyFlowToHost(flowTemp);
	flow->copyFlowColorToHost(flowTempRgb, 20.0f);

	clock_t timeElapsed = (clock() - start);

	cv::Mat uvrgb8, uvrgb8masked;
	flowTempRgb.convertTo(uvrgb8, CV_8UC3, 255.0f);
	uvrgb8.copyTo(uvrgb8masked, fisheyeMask8uc1);
	cv::putText(uvrgb8masked, std::to_string(timeElapsed) + " ms", cv::Point(10, 12),
		cv::FONT_HERSHEY_PLAIN, 1, CV_RGB(0, 0, 0));
	cv::imshow("flow", uvrgb8masked);

	flowTemp.copyTo(flowTempMasked, fisheyeMask8uc1);
	if (!flowTempMasked.isContinuous()) {
		flowTempMasked = flowTempMasked.clone();
	}
	flowTempMasked.reshape(2, flowTempMasked.cols * flowTempMasked.rows).copyTo(currMatchedPoints);

	return 0;
}