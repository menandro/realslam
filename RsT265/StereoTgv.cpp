#include "T265.h"

int T265::initStereoTGVL1() {
	stereotgv = new StereoTgv();
	//stereoScaling = 2.0f;
	// slow but okay settings
	/*float lambda = 3.0f;
	float nLevel = 10;
	float fScale = 1.2f;
	int nWarpIters = 50;
	int nSolverIters = 50;
	stereotgv->limitRange = 0.2f;*/
	// 20fps settings
	float lambda = 5.0f;
	float nLevel = 5;
	float fScale = 2.0f;
	int nWarpIters = 5;
	int nSolverIters = 20;
	stereotgv->limitRange = 0.5f;
	stereotgv->censusThreshold = 1.0f / 255.0f;

	//stereoWidth = (int)(this->width / stereoScaling);
	//stereoHeight = (int)(this->height / stereoScaling);
	stereotgv->baseline = 0.0642f;
	stereotgv->focal = 285.8557f / stereoScaling;

	this->Xraw = cv::Mat(stereoHeight, stereoWidth, CV_32FC3);
	this->X = cv::Mat::zeros(stereoHeight, stereoWidth, CV_32FC3);

	cv::Mat translationVector, calibrationVector;
	if (stereoScaling == 2.0f) {
		translationVector = cv::readOpticalFlow("translationVectorHalf.flo");
		calibrationVector = cv::readOpticalFlow("calibrationVectorHalf.flo");
	}
	else {
		translationVector = cv::readOpticalFlow("translationVector.flo");
		calibrationVector = cv::readOpticalFlow("calibrationVector.flo");
	}

	float beta = 4.0f;
	float gamma = 0.2f;
	float alpha0 = 17.0f; // 5.0f;
	float alpha1 = 1.2f;//  1.0f;
	float timeStepLambda = 1.0f;

	stereotgv->initialize(stereoWidth, stereoHeight, beta, gamma, alpha0, alpha1,
		timeStepLambda, lambda, nLevel, fScale, nWarpIters, nSolverIters);

	stereotgv->loadVectorFields(translationVector, calibrationVector);
	stereotgv->visualizeResults = true;

	// Load fisheye Mask
	if (stereoScaling == 2.0f) {
		this->fisheyeMask8 = cv::imread("maskHalf.png", cv::IMREAD_GRAYSCALE);
	}
	else {
		this->fisheyeMask8 = cv::imread("mask.png", cv::IMREAD_GRAYSCALE);
	}
	this->fisheyeMask8.convertTo(fisheyeMask, CV_32F, 1.0 / 255.0);
	stereotgv->copyMaskToDevice(fisheyeMask);
	return 0;
}

int T265::solveStereoTGVL1() {
	// Solve stereo depth
	cv::Mat equi1, equi2;
	cv::equalizeHist(this->fisheye1, equi1);
	cv::equalizeHist(this->fisheye2, equi2);

	if (stereoScaling == 2.0f) {
		cv::Mat halfFisheye1, halfFisheye2;
		//cv::resize(equi1, halfFisheye1, cv::Size(stereoWidth, stereoHeight));
		//cv::resize(equi2, halfFisheye2, cv::Size(stereoWidth, stereoHeight));
		cv::resize(this->fisheye1, halfFisheye1, cv::Size(stereoWidth, stereoHeight));
		cv::resize(this->fisheye2, halfFisheye2, cv::Size(stereoWidth, stereoHeight));

		clock_t start = clock();
		stereotgv->copyImagesToDevice(halfFisheye1, halfFisheye2);
		//stereotgv->solveStereoForwardMasked();
		stereotgv->solveStereoForwardCensusMasked();
		cv::Mat depth = cv::Mat(stereoHeight, stereoWidth, CV_32F);

		cv::Mat depthVis;
		//stereotgv->copyStereoToHost(depth);
		stereotgv->copyStereoToHost(depth, this->Xraw, 285.722f / stereoScaling, 286.759f / stereoScaling,
			420.135f / stereoScaling, 403.394 / stereoScaling,
			-0.00659769f, 0.0473251f, -0.0458264f, 0.00897725f,
			-0.0641854f, -0.000218299f, 0.000111253f);


		clock_t timeElapsed = (clock() - start);
		//std::cout << "time: " << timeElapsed << " ms" << std::endl;

		depth.copyTo(depth, fisheyeMask8);

		this->Xraw.copyTo(this->X, this->fisheyeMask8);
		//cv::imshow("X", pcXMasked);
		depth.copyTo(depthVis, fisheyeMask8);

		//cv::resize(depthVis, this->depth32f, cv::Size(this->width, this->height));
		showDepthJet("color", depthVis, std::to_string(timeElapsed), 5.0f, false);
		//cv::imshow("equi1", halfFisheye1);
	}
	else {
		clock_t start = clock();
		stereotgv->copyImagesToDevice(equi1, equi2);
		stereotgv->solveStereoForwardMasked();
		cv::Mat depth = cv::Mat(stereoHeight, stereoWidth, CV_32F);
		cv::Mat depthVis;

		stereotgv->copyStereoToHost(depth, this->Xraw, 285.722f / stereoScaling, 286.759f / stereoScaling,
			420.135f / stereoScaling, 403.394 / stereoScaling,
			-0.00659769f, 0.0473251f, -0.0458264f, 0.00897725f,
			-0.0641854f, -0.000218299f, 0.000111253f);
		clock_t timeElapsed = (clock() - start);

		depth.copyTo(depth, fisheyeMask8);

		depth.copyTo(depthVis, fisheyeMask8);
		showDepthJet("color", depthVis, std::to_string(timeElapsed), 5.0f, false);
		cv::imshow("equi1", equi1);
	}

	return 0;
}

void T265::showDepthJet(std::string windowName, cv::Mat image, float maxDepth, bool shouldWait = true) {
	cv::Mat u_norm, u_gray, u_color;
	u_norm = image * 256.0f / maxDepth;
	u_norm.convertTo(u_gray, CV_8UC1);
	cv::applyColorMap(u_gray, u_color, cv::COLORMAP_JET);

	cv::imshow(windowName, u_color);
	if (shouldWait) cv::waitKey();
}

void T265::showDepthJet(std::string windowName, cv::Mat image, std::string message, float maxDepth, bool shouldWait = true) {
	cv::Mat u_norm, u_gray, u_color;
	u_norm = image * 256.0f / maxDepth;
	u_norm.convertTo(u_gray, CV_8UC1);
	cv::applyColorMap(u_gray, u_color, cv::COLORMAP_JET);
	cv::putText(u_color, message, cv::Point(10, 12), cv::FONT_HERSHEY_PLAIN, 1, CV_RGB(255, 255, 255));
	cv::imshow(windowName, u_color);
	if (shouldWait) cv::waitKey();
}