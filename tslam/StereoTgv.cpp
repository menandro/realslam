#include "Tslam.h"

int Tslam::initStereoTGVL1() {
	stereotgv = new StereoTgv();
	stereoScaling = 2.0f;
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

	stereoWidth = (int)(t265.width / stereoScaling);
	stereoHeight = (int)(t265.height / stereoScaling);
	stereotgv->baseline = 0.0642f;
	stereotgv->focal = 285.8557f / stereoScaling;

	pcX = cv::Mat(stereoHeight, stereoWidth, CV_32FC3);
	pcXMasked = cv::Mat::zeros(stereoHeight, stereoWidth, CV_32FC3);

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

		clock_t start = clock();
		stereotgv->copyImagesToDevice(halfFisheye1, halfFisheye2);
		stereotgv->solveStereoForwardMasked();
		cv::Mat depth = cv::Mat(stereoHeight, stereoWidth, CV_32F);
		
		cv::Mat depthVis;
		stereotgv->copyStereoToHost(depth);
		stereotgv->copyStereoToHost(depth, pcX, 285.722f / stereoScaling, 286.759f / stereoScaling,
			420.135f / stereoScaling, 403.394 / stereoScaling,
			-0.00659769f, 0.0473251f, -0.0458264f, 0.00897725f,
			-0.0641854f, -0.000218299f, 0.000111253f);

		
		clock_t timeElapsed = (clock() - start);
		//std::cout << "time: " << timeElapsed << " ms" << std::endl;

		pcX.copyTo(pcXMasked, fisheyeMask8);
		//cv::imshow("X", pcXMasked);
		depth.copyTo(depthVis, fisheyeMask8);
		cv::resize(depthVis, t265.depth32f, cv::Size(t265.width, t265.height));
		showDepthJet("color", depthVis, std::to_string(timeElapsed), 5.0f, false);
		cv::imshow("equi1", halfFisheye1);
	}
	else {
		stereotgv->copyImagesToDevice(equi1, equi2);
		stereotgv->solveStereoForwardMasked();
		cv::Mat depth = cv::Mat(stereoHeight, stereoWidth, CV_32F);
		cv::Mat depthVis;
		stereotgv->copyStereoToHost(depth);
		depth.copyTo(depthVis, fisheyeMask8);
		showDepthJet("color", depthVis, 5.0f, false);
		cv::imshow("equi1", equi1);
	}

	return 0;
}

// Old TVL1, also wrong
//int Tslam::initStereoTVL1() {
//	stereo = new Stereo();
//	stereoScaling = 2.0f;
//	stereoWidth = (int)(t265.width / stereoScaling);
//	stereoHeight = (int)(t265.height / stereoScaling);
//	stereo->baseline = 0.0642f;
//	stereo->focal = 285.8557f / stereoScaling;
//	cv::Mat translationVector, calibrationVector;
//	if (stereoScaling == 2.0f) {
//		translationVector = cv::readOpticalFlow("translationVectorHalf.flo");
//		calibrationVector = cv::readOpticalFlow("calibrationVectorHalf.flo");
//	}
//	else {
//		translationVector = cv::readOpticalFlow("translationVector.flo");
//		calibrationVector = cv::readOpticalFlow("calibrationVector.flo");
//	}
//
//	//stereo->initializeFisheyeStereo(stereoWidth, stereoHeight, 1, CV_8U, 6, 2.0f, 50.0f, 0.33f, 0.125f, 1, 1000);
//	stereo->initializeFisheyeStereo(stereoWidth, stereoHeight, 1, CV_8U, 6, 2.0f, 50.0f, 0.33f, 0.125f, 1, 100);
//
//	stereo->loadVectorFields(translationVector, calibrationVector);
//	//stereo->initializeOpticalFlow(848, 800, 1, CV_8U, 6, 2.0f, 50.0f, 0.33f, 0.125f, 3, 200);
//	stereo->visualizeResults = true;
//	stereo->flowScale = 50.0f;
//	stereo->planeSweepMaxDisparity = 120;
//	stereo->planeSweepWindow = 5;
//	stereo->planeSweepMaxError = 0.05f;
//	stereo->planeSweepStride = 1;
//	stereo->isReverse = true;
//	return 0;
//}