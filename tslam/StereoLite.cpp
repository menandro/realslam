#include "Tslam.h"

int Tslam::initStereoTVL1() {
	stereolite = new StereoLite();
	stereoScaling = 2.0f;
	int nLevel = 5;
	float fScale = 2.0f;
	int nWarpIters = 10;
	int nSolverIters = 50;
	float lambda = 0.5f;
	float theta = 0.33f;
	float tau = 0.125f;
	stereolite->limitRange = 1.0f;

	// Planesweep settings
	stereolite->planeSweepMaxError = 1000.0f;
	stereolite->planeSweepMaxDisparity = (int)(50.0f / stereoScaling);
	stereolite->planeSweepStride = 0.5f;
	stereolite->planeSweepWindow = 11;
	stereolite->planeSweepEpsilon = 1.0f;

	// Planesweep + TVL1 settings
	stereolite->l2lambda = 0.1f;

	stereoWidth = (int)(t265.width / stereoScaling);
	stereoHeight = (int)(t265.height / stereoScaling);
	stereolite->baseline = 0.0642f;
	stereolite->focal = 285.8557f / stereoScaling;
	stereolite->initialize(stereoWidth, stereoHeight, lambda, theta, tau, nLevel, fScale, nWarpIters, nSolverIters);

	pcX = cv::Mat(stereoHeight, stereoWidth, CV_32FC3);
	pcXMasked = cv::Mat::zeros(stereoHeight, stereoWidth, CV_32FC3);

	// Load vector fields
	cv::Mat translationVector, calibrationVector;
	if (stereoScaling == 2.0f) {
		translationVector = cv::readOpticalFlow("../test_rstracking/translationVectorHalf.flo");
		calibrationVector = cv::readOpticalFlow("../test_rstracking/calibrationVectorHalf.flo");
	}
	else {
		translationVector = cv::readOpticalFlow("../test_rstracking/translationVector.flo");
		calibrationVector = cv::readOpticalFlow("../test_rstracking/calibrationVector.flo");
	}
	stereolite->loadVectorFields(translationVector, calibrationVector);

	// Load fisheye Mask
	fisheyeMask8Half = cv::imread("maskHalf.png", cv::IMREAD_GRAYSCALE);
	fisheyeMask8Half.convertTo(fisheyeMaskHalf, CV_32F, 1.0 / 255.0);
	fisheyeMask8 = cv::imread("mask.png", cv::IMREAD_GRAYSCALE);
	fisheyeMask8.convertTo(fisheyeMask, CV_32F, 1.0 / 255.0);
	if (stereoScaling == 2.0f) {
		stereolite->copyMaskToDevice(fisheyeMaskHalf);
	}
	else {
		stereolite->copyMaskToDevice(fisheyeMask);
	}

	return 0;
}

int Tslam::solveStereoTVL1() {
	// Solve stereo depth
	cv::Mat equi1, equi2;
	cv::equalizeHist(t265.fisheye1, equi1);
	cv::equalizeHist(t265.fisheye2, equi2);
	cv::Mat depth = cv::Mat(stereoHeight, stereoWidth, CV_32F);

	if (stereoScaling == 2.0f) {
		cv::Mat halfFisheye1, halfFisheye2;
		cv::resize(equi1, halfFisheye1, cv::Size(stereoWidth, stereoHeight));
		cv::resize(equi2, halfFisheye2, cv::Size(stereoWidth, stereoHeight));

		//cv::Mat depth = cv::Mat(stereoHeight, stereoWidth, CV_32F);
		clock_t start = clock();
		stereolite->copyImagesToDevice(halfFisheye1, halfFisheye2);

		// Planesweep with TVL1 refinement
		//stereolite->planeSweepPyramidL1Refinement();
		//stereolite->copyStereoToHost(t265.depthHalf32f);

		// Just planesweep
		//stereolite->planeSweepSubpixel();
		stereolite->planeSweepExponentialDistance();
		stereolite->copyPlanesweepFinalToHost(t265.depthHalf32f, pcX, 285.722f / stereoScaling, 286.759f / stereoScaling,
			420.135f / stereoScaling, 403.394 / stereoScaling,
			-0.00659769f, 0.0473251f, -0.0458264f, 0.00897725f,
			-0.0641854f, -0.000218299f, 0.000111253f);

		// Just TVL1
		//stereolite->solveStereoForwardMasked();
		//stereolite->copyStereoToHost(t265.depthHalf32f, pcX, 285.722f / stereoScaling, 286.759f / stereoScaling,
		//	420.135f / stereoScaling, 403.394 / stereoScaling,
		//	-0.00659769f, 0.0473251f, -0.0458264f, 0.00897725f,
		//	-0.0641854f, -0.000218299f, 0.000111253f);

		clock_t timeElapsed = (clock() - start);
		//std::cout << "time: " << timeElapsed << " ms" << std::endl;

		pcX.copyTo(pcXMasked, fisheyeMask8);
		cv::imshow("X", pcXMasked);

		cv::Mat depthVis;
		t265.depthHalf32f.copyTo(t265.depthHalf32f, fisheyeMask8Half);
		t265.depthHalf32f.copyTo(depthVis, fisheyeMask8Half);
		cv::resize(depthVis, t265.depth32f, cv::Size(t265.width, t265.height));
		showDepthJet("color", depthVis, std::to_string(timeElapsed), 5.0f, false);
		//showDepthJet("color", depthVis, "0", 5.0f, false);
		cv::imshow("equi1", halfFisheye1);

		// Convert to full-size image
		/*cv::Mat ddd;
		cv::resize(t265.depthHalf32f, t265.depth32f, cv::Size(t265.width, t265.height));
		t265.depth32f.setTo(0.0f, ~fisheyeMask8);
		t265.depth32f.copyTo(depthVis, fisheyeMask8);
		showDepthJet("color", depthVis, "0", 5.0f, false);
		cv::imshow("bw", t265.depth32f);*/
	}
	else {
		//cv::Mat depth = cv::Mat(stereoHeight, stereoWidth, CV_32F);
		stereolite->copyImagesToDevice(equi1, equi2);
		stereolite->planeSweepPyramidL1Refinement();
		stereolite->copyStereoToHost(t265.depth32f);
		cv::Mat depthVis;
		t265.depth32f.copyTo(depthVis, fisheyeMask8);
		showDepthJet("color", depthVis, 5.0f, false);
		cv::imshow("equi1", equi1);
	}

	return 0;
}