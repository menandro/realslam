#include "Tslam.h"

int Tslam::initStereoTGVL1() {
	stereotgv = new StereoTgv();
	stereoScaling = 2.0f;
	float lambda = 5.0f;
	float nLevel = 4;
	float fScale = 2.0f;
	int nWarpIters = 20;
	int nSolverIters = 20;

	stereoWidth = (int)(t265.width / stereoScaling);
	stereoHeight = (int)(t265.height / stereoScaling);
	stereotgv->baseline = 0.0642f;
	stereotgv->focal = 285.8557f / stereoScaling;
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
	float alpha0 = 5.0f;
	float alpha1 = 1.0f;
	float timeStepLambda = 1.0f;
	
	stereotgv->initialize(stereoWidth, stereoHeight, beta, gamma, alpha0, alpha1,
		timeStepLambda, lambda, nLevel, fScale, nWarpIters, nSolverIters);

	stereotgv->loadVectorFields(translationVector, calibrationVector);
	stereotgv->visualizeResults = true;

	// Load fisheye Mask
	if (stereoScaling == 2.0f) {
		fisheyeMask8 = cv::imread("maskHalf.png", cv::IMREAD_GRAYSCALE);
	}
	else {
		fisheyeMask8 = cv::imread("mask.png", cv::IMREAD_GRAYSCALE);
	}
	fisheyeMask8.convertTo(fisheyeMask, CV_32F, 1.0 / 255.0);
	stereotgv->copyMaskToDevice(fisheyeMask);
	return 0;
}

int Tslam::solveStereoTGVL1() {
	// Solve stereo depth
	cv::Mat equi1, equi2;
	cv::equalizeHist(t265.fisheye1, equi1);
	cv::equalizeHist(t265.fisheye2, equi2);

	if (stereoScaling == 2.0f) {
		cv::Mat halfFisheye1, halfFisheye2;
		cv::resize(equi1, halfFisheye1, cv::Size(stereoWidth, stereoHeight));
		cv::resize(equi2, halfFisheye2, cv::Size(stereoWidth, stereoHeight));
		stereotgv->copyImagesToDevice(halfFisheye1, halfFisheye2);
		stereotgv->solveStereoForwardMasked();
		cv::Mat depth = cv::Mat(stereoHeight, stereoWidth, CV_32F);
		cv::Mat depthVis;
		stereotgv->copyStereoToHost(depth);
		depth.copyTo(depthVis, fisheyeMask8);
		showDepthJet("color", depthVis, 5.0f, false);
	}
	else {
		stereotgv->copyImagesToDevice(equi1, equi2);
		stereotgv->solveStereoForwardMasked();
		cv::Mat depth = cv::Mat(stereoHeight, stereoWidth, CV_32F);
		cv::Mat depthVis;
		stereotgv->copyStereoToHost(depth);
		depth.copyTo(depthVis, fisheyeMask8);
		showDepthJet("color", depthVis, 5.0f, false);
	}

	return 0;
}

// Old TVL1, also wrong
int Tslam::initStereoTVL1() {
	stereo = new Stereo();
	stereoScaling = 2.0f;
	stereoWidth = (int)(t265.width / stereoScaling);
	stereoHeight = (int)(t265.height / stereoScaling);
	stereo->baseline = 0.0642f;
	stereo->focal = 285.8557f / stereoScaling;
	cv::Mat translationVector, calibrationVector;
	if (stereoScaling == 2.0f) {
		translationVector = cv::readOpticalFlow("translationVectorHalf.flo");
		calibrationVector = cv::readOpticalFlow("calibrationVectorHalf.flo");
	}
	else {
		translationVector = cv::readOpticalFlow("translationVector.flo");
		calibrationVector = cv::readOpticalFlow("calibrationVector.flo");
	}

	//stereo->initializeFisheyeStereo(stereoWidth, stereoHeight, 1, CV_8U, 6, 2.0f, 50.0f, 0.33f, 0.125f, 1, 1000);
	stereo->initializeFisheyeStereo(stereoWidth, stereoHeight, 1, CV_8U, 6, 2.0f, 50.0f, 0.33f, 0.125f, 1, 100);

	stereo->loadVectorFields(translationVector, calibrationVector);
	//stereo->initializeOpticalFlow(848, 800, 1, CV_8U, 6, 2.0f, 50.0f, 0.33f, 0.125f, 3, 200);
	stereo->visualizeResults = true;
	stereo->flowScale = 50.0f;
	stereo->planeSweepMaxDisparity = 120;
	stereo->planeSweepWindow = 5;
	stereo->planeSweepMaxError = 0.05f;
	stereo->planeSweepStride = 1;
	stereo->isReverse = true;
	return 0;
}