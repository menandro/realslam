#include "main.h"

int readCalibFileFaro(std::string filename, cv::Mat &K, cv::Mat &R, cv::Mat &t) {
	cv::FileStorage fs;
	fs.open(filename, cv::FileStorage::READ);
	fs["K"] >> K;
	fs["R"] >> R;
	fs["t"] >> t;
	return 0;
}

int test_censusTransform () {
	std::string mainfolder = "h:/data_rs_iis/20190909";
	int startFrame = 303;
	int endFrame = 303;

	StereoTgv* stereotgv = new StereoTgv();
	int width = 848;
	int height = 800;
	float stereoScaling = 1.0f;
	int nLevel = 12;
	float fScale = 1.2f;// 2.0f;
	int nWarpIters = 400;// 10;
	int nSolverIters = 100;// 20;
	float lambda = 1.5f;
	stereotgv->limitRange = 0.1f;
	float maxDepthVis = 10.0f;
	double sigmaBlur = 0.1;
	stereotgv->censusThreshold = 5.0f / 255.0f;

	int stereoWidth = (int)((float)width / stereoScaling);
	int stereoHeight = (int)((float)height / stereoScaling);
	stereotgv->baseline = 0.0642f;
	stereotgv->focal = 285.8557f / stereoScaling;
	cv::Mat translationVector, calibrationVector;
	if (stereoScaling == 2.0f) {
		translationVector = cv::readOpticalFlow("h:/data_rs_iis/translationVectorHalf.flo");
		calibrationVector = cv::readOpticalFlow("h:/data_rs_iis/calibrationVectorHalf.flo");
	}
	else {
		translationVector = cv::readOpticalFlow("h:/data_rs_iis/translationVector.flo");
		calibrationVector = cv::readOpticalFlow("h:/data_rs_iis/calibrationVector.flo");
	}

	float beta = 4.0f;
	float gamma = 0.2f;
	float alpha0 = 17.0f;
	float alpha1 = 1.2f;
	float timeStepLambda = 1.0f;

	stereotgv->initialize(stereoWidth, stereoHeight, beta, gamma, alpha0, alpha1,
		timeStepLambda, lambda, nLevel, fScale, nWarpIters, nSolverIters);
	stereotgv->visualizeResults = true;

	cv::Mat mask = cv::Mat::zeros(cv::Size(stereoWidth, stereoHeight), CV_8UC1);
	circle(mask, cv::Point(stereoWidth / 2, stereoHeight / 2), stereoWidth / 2 - (int)(40 / stereoScaling), cv::Scalar(256.0f), -1);
	cv::Mat fisheyeMask;
	mask.convertTo(fisheyeMask, CV_32F, 1.0 / 255.0);

	stereotgv->copyMaskToDevice(fisheyeMask);

	// Load vector fields
	stereotgv->loadVectorFields(translationVector, calibrationVector);

	cv::Mat halfFisheye1, halfFisheye2;
	cv::Mat equi1, equi2;

	for (int k = startFrame; k <= endFrame; k++) {
		cv::Mat im1 = cv::imread(mainfolder + "/colored_0/data/im" + std::to_string(k) + ".png", cv::IMREAD_GRAYSCALE);
		cv::Mat im2 = cv::imread(mainfolder + "/colored_1/data/im" + std::to_string(k) + ".png", cv::IMREAD_GRAYSCALE);

		cv::equalizeHist(im1, equi1);
		cv::equalizeHist(im2, equi2);
		cv::resize(im1, halfFisheye1, cv::Size(stereoWidth, stereoHeight));
		cv::resize(im2, halfFisheye2, cv::Size(stereoWidth, stereoHeight));
		clock_t start = clock();

		stereotgv->copyImagesToDevice(halfFisheye1, halfFisheye2);
		//stereotgv->solveStereoForwardMasked();
		stereotgv->solveStereoForwardCensusMasked();

		cv::Mat depth = cv::Mat(stereoHeight, stereoWidth, CV_32F);
		cv::Mat depthVis;
		cv::Mat Xraw = cv::Mat(stereoHeight, stereoWidth, CV_32FC3);
		stereotgv->copyStereoToHost(depth, Xraw, 285.722f / stereoScaling, 286.759f / stereoScaling,
			420.135f / stereoScaling, 403.394 / stereoScaling,
			-0.00659769f, 0.0473251f, -0.0458264f, 0.00897725f,
			-0.0641854f, -0.000218299f, 0.000111253f);

		clock_t timeElapsed = (clock() - start);
		std::cout << "time: " << timeElapsed << " ms" << std::endl;

		depth.copyTo(depthVis, mask);
		showDepthJetExponential("color", depthVis, maxDepthVis, 0.5f, false);

		std::string appender;
		if (k < 10) appender = "000";
		else if ((k >= 10) && (k < 100)) appender = "00";
		else if ((k >= 100) && (k < 1000)) appender = "0";
		else appender = "";

		cv::Mat depth16;
		depthVis.convertTo(depth16, CV_16U, 256.0f);
		cv::imwrite(mainfolder + "/outputcensus/im" + appender + std::to_string(k) + ".png", depth16);
		saveDepthJetExponential(mainfolder + "/error/census/im" + appender + std::to_string(k) + ".png",
			depthVis, maxDepthVis, 0.5f);

		std::cout << k << std::endl;
		//cv::imshow("test", halfFisheye1);
		cv::waitKey(1);
	}
	cv::waitKey();
	return 0;
}

int test_BlenderPerspectiveDataSequence() {
	std::string folder = "h:/blender/pers01";
	cv::Mat translationVector = cv::readOpticalFlow(folder + "/translationVectorBlenderPerspective.flo");
	int stereoWidth = 800;
	int stereoHeight = 800;
	cv::Mat calibrationVector = cv::Mat::zeros(cv::Size(stereoWidth, stereoHeight), CV_32FC2);
	cv::Mat mask = cv::Mat::zeros(cv::Size(stereoWidth, stereoHeight), CV_8UC1) + 255;
	cv::Mat fisheyeMask;
	mask.convertTo(fisheyeMask, CV_32F, 1.0 / 255.0);

	StereoTgv * stereotgv = new StereoTgv();
	int width = 800;
	int height = 800;
	float stereoScaling = 1.0f;
	int nLevel = 11; // 11 in paper
	float fScale = 1.2f;
	int nWarpIters = 50;
	int nSolverIters = 50;
	float lambda = 5.0f;
	stereotgv->limitRange = 1.0f;


	float beta = 9.0f;
	float gamma = 0.85f;
	float alpha0 = 17.0f;
	float alpha1 = 1.2f;
	float timeStepLambda = 1.0f;

	stereotgv->initialize(stereoWidth, stereoHeight, beta, gamma, alpha0, alpha1,
		timeStepLambda, lambda, nLevel, fScale, nWarpIters, nSolverIters);
	stereotgv->visualizeResults = true;

	stereotgv->copyMaskToDevice(fisheyeMask);
	stereotgv->loadVectorFields(translationVector, calibrationVector);
	cv::Mat equi1, equi2;

	for (int k = 1; k <= 300; k++) {
		std::string appender;
		if (k < 10) appender = "000";
		else if ((k >= 10) && (k < 100)) appender = "00";
		else if ((k >= 100) && (k < 1000)) appender = "0";
		else appender = "";
		std::string outputFilename = folder + "/video_output2/" + appender + std::to_string(k) + ".flo";
		cv::Mat im1 = cv::imread(folder + "/left/image/Image" + appender + std::to_string(k) + ".png", cv::IMREAD_GRAYSCALE);
		cv::Mat im2 = cv::imread(folder + "/right2/image/Image" + appender + std::to_string(k) + ".png", cv::IMREAD_GRAYSCALE);

		cv::equalizeHist(im1, equi1);
		cv::equalizeHist(im2, equi2);
		//stereotgv->copyImagesToDevice(im1, im2);
		stereotgv->copyImagesToDevice(equi1, equi2);
		stereotgv->solveStereoForwardMasked();

		cv::Mat disparity = cv::Mat(stereoHeight, stereoWidth, CV_32FC2);
		stereotgv->copyDisparityToHost(disparity);
		cv::writeOpticalFlow(outputFilename, disparity);

		cv::imshow("left", im1);
		cv::waitKey(1);
		std::cout << k << std::endl;
	}
	cv::waitKey();
}

int test_SolveTrajectoryPerWarpingBlenderDataAll() {
	std::string folder = "h:/blender/icra2020";
	cv::Mat translationVector = cv::readOpticalFlow(folder + "/translationVectorBlender.flo");
	int stereoWidth = 800;
	int stereoHeight = 800;
	cv::Mat calibrationVector = cv::Mat::zeros(cv::Size(stereoWidth, stereoHeight), CV_32FC2);
	cv::Mat mask = cv::Mat::zeros(cv::Size(stereoWidth, stereoHeight), CV_8UC1);
	circle(mask, cv::Point(stereoWidth / 2, stereoHeight / 2), stereoWidth / 2 - 50, cv::Scalar(256.0f), -1);
	cv::Mat fisheyeMask;
	mask.convertTo(fisheyeMask, CV_32F, 1.0 / 255.0);

	float cx = 400.0f;
	float cy = 400.0f;
	float focal = 800.0f / (3.14159265359f);
	float fov = 3.14159265359f;
	float tx = -0.06f;
	float ty = 0.0f;
	float tz = 0.0f;

	StereoTgv * stereotgv = new StereoTgv();
	int width = 800;
	int height = 800;
	float stereoScaling = 1.0f;
	int nLevel = 11; // 11 in paper
	float fScale = 1.2f;
	int nWarpIters = 50;
	int nSolverIters = 50;
	float lambda = 5.0f;
	stereotgv->limitRange = 0.1f;


	float beta = 9.0f;
	float gamma = 0.85f;
	float alpha0 = 17.0f;
	float alpha1 = 1.2f;
	float timeStepLambda = 1.0f;

	stereotgv->initialize(stereoWidth, stereoHeight, beta, gamma, alpha0, alpha1,
		timeStepLambda, lambda, nLevel, fScale, nWarpIters, nSolverIters);
	stereotgv->visualizeResults = true;

	stereotgv->copyMaskToDevice(fisheyeMask);
	stereotgv->loadVectorFields(translationVector, calibrationVector);
	cv::Mat equi1, equi2;

	for (int k = 1; k <= 300; k++) {
		std::string appender;
		if (k < 10) appender = "000";
		else if ((k >= 10) && (k < 100)) appender = "00";
		else if ((k >= 100) && (k < 1000)) appender = "0";
		else appender = "";
		std::string outputFilename = folder + "/video_outputactualcurve/" + appender + std::to_string(k) + ".flo";
		cv::Mat im1 = cv::imread(folder + "/left/" + appender + std::to_string(k) + ".png", cv::IMREAD_GRAYSCALE);
		cv::Mat im2 = cv::imread(folder + "/right/" + appender + std::to_string(k) + ".png", cv::IMREAD_GRAYSCALE);

		cv::equalizeHist(im1, equi1);
		cv::equalizeHist(im2, equi2);
		//stereotgv->copyImagesToDevice(im1, im2);
		stereotgv->copyImagesToDevice(equi1, equi2);
		stereotgv->solveStereoTrajectoryPerIteration(fov, cx, cy, tx, ty, tz);

		cv::Mat disparity = cv::Mat(stereoHeight, stereoWidth, CV_32FC2);
		stereotgv->copyDisparityToHost(disparity);
		cv::writeOpticalFlow(outputFilename, disparity);

		cv::imshow("left", im1);
		cv::waitKey(1);
		std::cout << k << std::endl;
	}
}

int test_SolveTrajectoryPerWarpingBlenderData() {
	//std::string folder = "C:/Users/cvl-menandro/Downloads/rpg_urban_blender.tar/rpg_urban_blender";
	std::string folder = "h:/blender/icra2020";
	std::string outputFilename = folder + "/outputactualcurve/output11.flo";
	cv::Mat im1 = cv::imread(folder + "/image/left11.png", cv::IMREAD_GRAYSCALE);
	cv::Mat im2 = cv::imread(folder + "/image/right11.png", cv::IMREAD_GRAYSCALE);
	cv::Mat translationVector = cv::readOpticalFlow(folder + "/translationVectorBlender.flo");
	float cx = 400.0f;
	float cy = 400.0f;
	float focal = 800.0f / (3.14159265359f);
	float fov = 3.14159265359f;
	float tx = -0.06f;
	float ty = 0.0f;
	float tz = 0.0f;

	int stereoWidth = im1.cols;
	int stereoHeight = im1.rows;

	cv::Mat calibrationVector = cv::Mat::zeros(cv::Size(stereoWidth, stereoHeight), CV_32FC2);
	cv::Mat mask = cv::Mat::zeros(cv::Size(stereoWidth, stereoHeight), CV_8UC1);
	circle(mask, cv::Point(stereoWidth / 2, stereoHeight / 2), stereoWidth / 2 - 50, cv::Scalar(256.0f), -1);
	//cv::imwrite("maskBlender.png", mask);
	//return 0;
	/*cv::imshow("mask", mask);
	std::cout << (int)mask.at<unsigned char>(400, 400) << std::endl;
	cv::waitKey();*/
	cv::Mat fisheyeMask;
	mask.convertTo(fisheyeMask, CV_32F, 1.0 / 255.0);

	StereoTgv * stereotgv = new StereoTgv();
	int width = 800;
	int height = 800;
	float stereoScaling = 1.0f;
	int nLevel = 11; // 11 in paper
	float fScale = 1.2f;
	int nWarpIters = 50;
	int nSolverIters = 50;
	float lambda = 5.0f;
	stereotgv->limitRange = 0.1f;

	float beta = 9.0f;
	float gamma = 0.85f;
	float alpha0 = 17.0f;
	float alpha1 = 1.2f;
	float timeStepLambda = 1.0f;

	stereotgv->initialize(stereoWidth, stereoHeight, beta, gamma, alpha0, alpha1,
		timeStepLambda, lambda, nLevel, fScale, nWarpIters, nSolverIters);
	stereotgv->visualizeResults = true;

	stereotgv->copyMaskToDevice(fisheyeMask);
	stereotgv->loadVectorFields(translationVector, calibrationVector);
	cv::Mat equi1, equi2;
	cv::equalizeHist(im1, equi1);
	cv::equalizeHist(im2, equi2);
	//stereotgv->copyImagesToDevice(im1, im2);
	stereotgv->copyImagesToDevice(equi1, equi2);
	stereotgv->solveStereoTrajectoryPerIteration(fov, cx, cy, tx, ty, tz);

	cv::Mat disparityVis = cv::Mat(stereoHeight, stereoWidth, CV_32FC3);
	stereotgv->copyDisparityVisToHost(disparityVis, 50.0f);
	cv::imshow("flow", disparityVis);

	cv::Mat disparity = cv::Mat(stereoHeight, stereoWidth, CV_32FC2);
	stereotgv->copyDisparityToHost(disparity);
	cv::writeOpticalFlow(outputFilename, disparity);
	// convert disparity to 3D (depends on the model)

	cv::Mat depthVis;
	cv::Mat depth = cv::Mat(stereoHeight, stereoWidth, CV_32F);
	stereotgv->copyStereoToHost(depth);
	depth.copyTo(depthVis, mask);
	showDepthJet("color", depthVis, 5.0f, false);

	cv::Mat warped;
	stereotgv->copyWarpedImageToHost(warped);
	cv::imshow("left", im1);
	cv::imshow("right", im2);
	cv::imshow("warped", warped);
	cv::waitKey();
}

int test_SolveTrajectoryPerWarpingFaro(){
	std::string folder = "h:/data_icra/";

	StereoTgv * stereotgv = new StereoTgv();
	int width = 800;
	int height = 800;
	float stereoScaling = 1.0f;
	int nLevel = 11;
	float fScale = 1.2f;
	int nWarpIters = 500;
	int nSolverIters = 50;
	float lambda = 5.0f;
	stereotgv->limitRange = 0.1f;
	float beta = 9.0f;//4.0f;
	float gamma = 0.85f;// 0.2f;
	float alpha0 = 17.0f;// 5.0f;
	float alpha1 = 1.2f;// 1.0f;
	float timeStepLambda = 1.0f;

	stereotgv->initialize(width, height, beta, gamma, alpha0, alpha1,
		timeStepLambda, lambda, nLevel, fScale, nWarpIters, nSolverIters);
	stereotgv->visualizeResults = true;

	cv::Mat mask = cv::Mat::zeros(cv::Size(width, height), CV_8UC1);
	circle(mask, cv::Point(width / 2, height / 2), width / 2 - 10, cv::Scalar(256.0f), -1);
	cv::Mat fisheyeMask;
	mask.convertTo(fisheyeMask, CV_32F, 1.0 / 255.0);
	stereotgv->copyMaskToDevice(fisheyeMask);

	int list[18] = {114, 117, 124, 132, 140, 148, 157, 165, 172, 181, 187, 196, 203, 205, 211, 213, 219, 221};
	for (int counter = 0; counter < 1 ; counter++) {
		int k = 203;// list[counter];
		std::string filename = "im" + std::to_string(k);
		std::string outputFilename = folder + "outputactualcurve/" + filename + ".flo";
		cv::Mat im1 = cv::imread(folder + "image_02/data/" + filename + ".png");
		cv::Mat im2 = cv::imread(folder + "image_03/data/" + filename + ".png");
		cv::Mat K, R, t;
		readCalibFileFaro(folder + "calib/" + filename + ".xml", K, R, t);
		float cx = 400.0f;
		float cy = 400.0f;
		float focal = 800.0f / (2.0f * 3.14159265359f / 3.0f);
		float fov = (2.0f * 3.14159265359f / 3.0f);
		float tx = (float)t.at<double>(0);
		float ty = (float)t.at<double>(1);
		float tz = (float)t.at<double>(2);
		//std::cout << cx << " " << cy << " " << focal << " " << tx << " " << ty << " " << tz << std::endl;

		cv::Mat equi1, equi2;
		cv::cvtColor(im1, im1, cv::COLOR_BGR2GRAY);
		cv::cvtColor(im2, im2, cv::COLOR_BGR2GRAY);
		cv::equalizeHist(im1, equi1);
		cv::equalizeHist(im2, equi2);
		int stereoWidth = width;
		int stereoHeight = height;
		cv::Mat translationVector = cv::readOpticalFlow(folder + "translationVector/" + filename + ".flo");
		//cv::Mat translationVector = cv::Mat::zeros(im1.size(), CV_32FC2);
		cv::Mat calibrationVector = cv::readOpticalFlow(folder + "calibrationVector/" + filename + ".flo");

		stereotgv->loadVectorFields(translationVector, calibrationVector);

		clock_t start = clock();

		stereotgv->copyImagesToDevice(equi1, equi2);
		stereotgv->solveStereoTrajectoryPerIteration(fov, cx, cy, tx, ty, tz);

		cv::Mat disparityVis = cv::Mat(stereoHeight, stereoWidth, CV_32FC3);
		stereotgv->copyDisparityVisToHost(disparityVis, 50.0f);
		cv::imshow("flow", disparityVis);

		clock_t timeElapsed = (clock() - start);
		std::cout << "actual time: " << timeElapsed << " ms" << std::endl;

		cv::Mat disparity = cv::Mat(stereoHeight, stereoWidth, CV_32FC2);
		stereotgv->copyDisparityToHost(disparity);
		cv::writeOpticalFlow(outputFilename, disparity);
		// convert disparity to 3D (depends on the model)

		cv::Mat depthVis;
		cv::Mat depth = cv::Mat(stereoHeight, stereoWidth, CV_32F);
		stereotgv->copyStereoToHost(depth);
		depth.copyTo(depthVis, mask);
		//showDepthJet("color", depthVis, 5.0f, false);

		cv::Mat warped;
		stereotgv->copyWarpedImageToHost(warped);
		//cv::imshow("right", equi2);
		//cv::imshow("left", equi1);
		cv::imshow("warped", warped);
		cv::waitKey(1);
	}
	cv::waitKey();
	return 0;
}

int test_BlenderDataAllPlanesweep() {
	std::string folder = "h:/blender/icra2020/";
	StereoLite * stereotgv = new StereoLite();
	int width = 800;
	int height = 800;
	int nLevel = 5;
	float fScale = 2.0f;
	int nWarpIters = 10;
	int nSolverIters = 5;
	float lambda = 1.0f;
	float theta = 0.33f;
	float tau = 0.125f;
	stereotgv->limitRange = 0.2f;
	stereotgv->planeSweepMaxError = 1000.0f;
	stereotgv->planeSweepMaxDisparity = (int)(50.0f);
	stereotgv->planeSweepStride = 0.01f;
	stereotgv->planeSweepWindow = 11;
	stereotgv->planeSweepEpsilon = 1.0f;
	//int upsamplingRadius = 5;
	stereotgv->maxPropIter = 3;
	stereotgv->l2lambda = 1.0f;
	stereotgv->l2iter = 500;
	stereotgv->planeSweepStandardDev = 1.0f;

	int stereoWidth = width;
	int stereoHeight = height;
	stereotgv->baseline = 0.0642f;
	stereotgv->focal = 285.8557f;

	stereotgv->initialize(stereoWidth, stereoHeight, lambda, theta, tau, nLevel, fScale, nWarpIters, nSolverIters);

	// Load fisheye Mask
	cv::Mat mask = cv::Mat::zeros(cv::Size(stereoWidth, stereoHeight), CV_8UC1);
	circle(mask, cv::Point(stereoWidth / 2, stereoHeight / 2), stereoWidth / 2 - 50, cv::Scalar(256.0f), -1);
	cv::Mat fisheyeMask;
	mask.convertTo(fisheyeMask, CV_32F, 1.0 / 255.0);
	stereotgv->copyMaskToDevice(fisheyeMask);

	// Load vector fields
	cv::Mat translationVector = cv::readOpticalFlow(folder + "/translationVectorBlender.flo");
	cv::Mat calibrationVector = cv::Mat::zeros(cv::Size(stereoWidth, stereoHeight), CV_32FC2);
	stereotgv->loadVectorFields(translationVector, calibrationVector);

	cv::Mat halfFisheye1, halfFisheye2;
	cv::Mat equi1, equi2;

	clock_t avetime = 0;
	cv::Mat disparity = cv::Mat(stereoHeight, stereoWidth, CV_32FC2);

	for (int k = 1; k <= 300; k++) {
		std::string appender;
		if (k < 10) appender = "000";
		else if ((k >= 10) && (k < 100)) appender = "00";
		else if ((k >= 100) && (k < 1000)) appender = "0";
		else appender = "";
		std::string outputFilename = folder + "/video_outputplanesweep/" + appender + std::to_string(k) + ".flo";
		cv::Mat im1 = cv::imread(folder + "/left/" + appender + std::to_string(k) + ".png", cv::IMREAD_GRAYSCALE);
		cv::Mat im2 = cv::imread(folder + "/right/" + appender + std::to_string(k) + ".png", cv::IMREAD_GRAYSCALE);

		// Solve stereo depth
		cv::equalizeHist(im1, equi1);
		cv::equalizeHist(im2, equi2);

		stereotgv->copyImagesToDevice(equi1, equi2);
		//stereotgv->planeSweep();
		stereotgv->planeSweepSubpixel();

		cv::Mat disparityVis = cv::Mat(stereoHeight, stereoWidth, CV_32FC3);
		stereotgv->copyPlanesweepDisparityVisToHost(disparityVis, 10.0f);
		cv::imshow("flow", disparityVis);

		stereotgv->copyPlanesweepDisparityToHost(disparity);
		cv::writeOpticalFlow(outputFilename, disparity);

		cv::waitKey(1);
	}
	cv::waitKey();
	return 0;
}

int test_FaroDataAllPlanesweep() {
	std::string folder = "h:/data_icra/";
	StereoLite * stereotgv = new StereoLite();
	int width = 800;
	int height = 800;
	int nLevel = 5;
	float fScale = 2.0f;
	int nWarpIters = 10;
	int nSolverIters = 5;
	float lambda = 1.0f;
	float theta = 0.33f;
	float tau = 0.125f;
	stereotgv->limitRange = 0.2f;
	stereotgv->planeSweepMaxError = 1000.0f;
	stereotgv->planeSweepMaxDisparity = (int)(50.0f);
	stereotgv->planeSweepStride = 0.05f;
	stereotgv->planeSweepWindow = 11;
	stereotgv->planeSweepEpsilon = 0.5f;
	//int upsamplingRadius = 5;
	stereotgv->maxPropIter = 3;
	stereotgv->l2lambda = 1.0f;
	stereotgv->l2iter = 500;
	stereotgv->planeSweepStandardDev = 1.0f;

	int stereoWidth = width;
	int stereoHeight = height;
	stereotgv->baseline = 0.0642f;
	stereotgv->focal = 285.8557f;
	cv::Mat translationVector, calibrationVector;
	
	stereotgv->initialize(stereoWidth, stereoHeight, lambda, theta, tau, nLevel, fScale, nWarpIters, nSolverIters);

	// Load fisheye Mask
	cv::Mat mask = cv::Mat::zeros(cv::Size(width, height), CV_8UC1);
	circle(mask, cv::Point(width / 2, height / 2), width / 2 - 10, cv::Scalar(256.0f), -1);
	cv::Mat fisheyeMask;
	mask.convertTo(fisheyeMask, CV_32F, 1.0 / 255.0);
	stereotgv->copyMaskToDevice(fisheyeMask);

	cv::Mat halfFisheye1, halfFisheye2;
	cv::Mat equi1, equi2;

	clock_t avetime = 0;

	for (int k = 145; k <= 224; k++) {
		std::string filename = "im" + std::to_string(k);
		std::string outputFilename = folder + "outputplanesweep/" + filename + ".flo";
		cv::Mat im1 = cv::imread(folder + "image_02/data/" + filename + ".png");
		cv::Mat im2 = cv::imread(folder + "image_03/data/" + filename + ".png");
		cv::Mat equi1, equi2;
		cv::cvtColor(im1, im1, cv::COLOR_BGR2GRAY);
		cv::cvtColor(im2, im2, cv::COLOR_BGR2GRAY);
		cv::equalizeHist(im1, equi1);
		cv::equalizeHist(im2, equi2);
		int stereoWidth = width;
		int stereoHeight = height;
		cv::Mat translationVector = cv::readOpticalFlow(folder + "translationVector/" + filename + ".flo");
		cv::Mat calibrationVector = cv::readOpticalFlow(folder + "calibrationVector/" + filename + ".flo");

		// Load vector fields
		stereotgv->loadVectorFields(translationVector, calibrationVector);

		// Solve stereo depth
		cv::equalizeHist(im1, equi1);
		cv::equalizeHist(im2, equi2);

		cv::Mat disparity = cv::Mat(stereoHeight, stereoWidth, CV_32FC2);

		stereotgv->copyImagesToDevice(equi1, equi2);
		//stereotgv->planeSweep();
		stereotgv->planeSweepSubpixel();
		
		cv::Mat disparityVis = cv::Mat(stereoHeight, stereoWidth, CV_32FC3);
		stereotgv->copyPlanesweepDisparityVisToHost(disparityVis, 50.0f);
		cv::imshow("flow", disparityVis);

		stereotgv->copyPlanesweepDisparityToHost(disparity);
		cv::writeOpticalFlow(outputFilename, disparity);

		cv::waitKey(1);
	}
	cv::waitKey();
	return 0;
}

int test_FaroDataAll() {
	//std::string folder = "C:/Users/cvl-menandro/Downloads/rpg_urban_blender.tar/rpg_urban_blender";
	std::string folder = "h:/data_icra/";
	
	StereoTgv * stereotgv = new StereoTgv();
	int width = 800;
	int height = 800;
	float stereoScaling = 1.0f;
	int nLevel = 11;
	float fScale = 1.2f;
	int nWarpIters = 50;
	int nSolverIters = 50;
	float lambda = 5.0f;
	stereotgv->limitRange = 0.1f;
	float beta = 9.0f;//4.0f;
	float gamma = 0.85f;// 0.2f;
	float alpha0 = 17.0f;// 5.0f;
	float alpha1 = 1.2f;// 1.0f;
	float timeStepLambda = 1.0f;

	stereotgv->initialize(width, height, beta, gamma, alpha0, alpha1,
		timeStepLambda, lambda, nLevel, fScale, nWarpIters, nSolverIters);
	stereotgv->visualizeResults = true;

	cv::Mat mask = cv::Mat::zeros(cv::Size(width, height), CV_8UC1);
	circle(mask, cv::Point(width / 2, height / 2), width / 2 - 10, cv::Scalar(256.0f), -1);
	cv::Mat fisheyeMask;
	mask.convertTo(fisheyeMask, CV_32F, 1.0 / 255.0);
	stereotgv->copyMaskToDevice(fisheyeMask);

	for (int k = 81; k <= 224; k++) {
		std::string filename = "im" + std::to_string(k);
		std::string outputFilename = folder + "output/" + filename + ".flo";
		cv::Mat im1 = cv::imread(folder + "image_02/data/" + filename + ".png");
		cv::Mat im2 = cv::imread(folder + "image_03/data/" + filename + ".png");
		cv::Mat equi1, equi2;
		cv::cvtColor(im1, im1, cv::COLOR_BGR2GRAY);
		cv::cvtColor(im2, im2, cv::COLOR_BGR2GRAY);
		cv::equalizeHist(im1, equi1);
		cv::equalizeHist(im2, equi2);
		int stereoWidth = width;
		int stereoHeight = height;
		cv::Mat translationVector = cv::readOpticalFlow(folder + "translationVector/" + filename + ".flo");
		cv::Mat calibrationVector = cv::readOpticalFlow(folder + "calibrationVector/" + filename + ".flo");
		
		stereotgv->loadVectorFields(translationVector, calibrationVector);

		clock_t start = clock();

		stereotgv->copyImagesToDevice(equi1, equi2);
		stereotgv->solveStereoForwardMasked();

		cv::Mat disparityVis = cv::Mat(stereoHeight, stereoWidth, CV_32FC3);
		stereotgv->copyDisparityVisToHost(disparityVis, 50.0f);
		cv::imshow("flow", disparityVis);

		clock_t timeElapsed = (clock() - start);
		std::cout << "ours time: " << timeElapsed << " ms" << std::endl;

		cv::Mat disparity = cv::Mat(stereoHeight, stereoWidth, CV_32FC2);
		stereotgv->copyDisparityToHost(disparity);
		cv::writeOpticalFlow(outputFilename, disparity);
		// convert disparity to 3D (depends on the model)

		cv::Mat depthVis;
		cv::Mat depth = cv::Mat(stereoHeight, stereoWidth, CV_32F);
		stereotgv->copyStereoToHost(depth);
		depth.copyTo(depthVis, mask);
		//showDepthJet("color", depthVis, 5.0f, false);

		cv::Mat warped;
		stereotgv->copyWarpedImageToHost(warped);
		//cv::imshow("right", equi2);
		//cv::imshow("left", equi1);
		cv::imshow("warped", warped);
		cv::waitKey(1);
	}
	return 0;
}

int test_VehicleSegmentationSequence() {
	std::string mainfolder = "h:/data_rs_iis/20190913_1";
	std::string outputfolder = "/segmented";
	StereoLite * stereotgv = new StereoLite();
	int width = 848;
	int height = 800;
	float stereoScaling = 2.0f;
	int nLevel = 5;
	float fScale = 2.0f;
	int nWarpIters = 10;
	int nSolverIters = 5;
	float lambda = 1.0f;
	float theta = 0.33f;
	float tau = 0.125f;
	stereotgv->limitRange = 0.2f;
	stereotgv->planeSweepMaxError = 1000.0f;
	stereotgv->planeSweepMaxDisparity = (int)(50.0f / stereoScaling);
	stereotgv->planeSweepStride = 1;
	stereotgv->planeSweepWindow = 9;
	stereotgv->planeSweepEpsilon = 1.0f;
	stereotgv->l2lambda = 1.0f;

	int stereoWidth = (int)((float)width / stereoScaling);
	int stereoHeight = (int)((float)height / stereoScaling);
	stereotgv->baseline = 0.0642f;
	stereotgv->focal = 285.8557f / stereoScaling;
	cv::Mat translationVector, calibrationVector;
	if (stereoScaling == 2.0f) {
		translationVector = cv::readOpticalFlow("../test_rstracking/translationVectorHalf.flo");
		calibrationVector = cv::readOpticalFlow("../test_rstracking/calibrationVectorHalf.flo");
	}
	else {
		translationVector = cv::readOpticalFlow("../test_rstracking/translationVector.flo");
		calibrationVector = cv::readOpticalFlow("../test_rstracking/calibrationVector.flo");
	}

	stereotgv->initialize(stereoWidth, stereoHeight, lambda, theta, tau, nLevel, fScale, nWarpIters, nSolverIters);

	// Load fisheye Mask
	cv::Mat fisheyeMask8, fisheyeMask;
	if (stereoScaling == 2.0f) fisheyeMask8 = cv::imread("maskHalf.png", cv::IMREAD_GRAYSCALE);
	else fisheyeMask8 = cv::imread("mask.png", cv::IMREAD_GRAYSCALE);
	fisheyeMask8.convertTo(fisheyeMask, CV_32F, 1.0 / 255.0);
	stereotgv->copyMaskToDevice(fisheyeMask);

	// Load vector fields
	stereotgv->loadVectorFields(translationVector, calibrationVector);

	cv::Mat halfFisheye1, halfFisheye2;
	cv::Mat equi1, equi2;

	clock_t avetime = 0;

	for (int k = 0; k <= 1274; k++) {
		cv::Mat im1 = cv::imread(mainfolder + "/colored_0/data/im" + std::to_string(k) + ".png", cv::IMREAD_GRAYSCALE);
		cv::Mat im2 = cv::imread(mainfolder + "/colored_1/data/im" + std::to_string(k) + ".png", cv::IMREAD_GRAYSCALE);

		// Solve stereo depth
		cv::equalizeHist(im1, equi1);
		cv::equalizeHist(im2, equi2);
		cv::resize(equi1, halfFisheye1, cv::Size(stereoWidth, stereoHeight));
		cv::resize(equi2, halfFisheye2, cv::Size(stereoWidth, stereoHeight));

		cv::Mat depth = cv::Mat(stereoHeight, stereoWidth, CV_32F);
		cv::Mat depthPs = cv::Mat(stereoHeight, stereoWidth, CV_32F);

		clock_t start = clock();
		stereotgv->copyImagesToDevice(halfFisheye1, halfFisheye2);
		stereotgv->planeSweepPyramidL1Refinement();
		stereotgv->copyPlanesweepFinalToHost(depth);
		depthPs = depth.clone();
		stereotgv->copyStereoToHost(depth);
		clock_t timeElapsed = (clock() - start);
		avetime = (avetime + timeElapsed) / 2;
		std::cout << "time: " << avetime << " ms ";

		cv::Mat depthVis;
		depth.copyTo(depthVis, fisheyeMask8);
		showDepthJet("color", depthVis, 5.0f, false);

		cv::Mat depthVisPs;
		depthPs.copyTo(depthVisPs, fisheyeMask8);
		showDepthJet("ps", depthVisPs, 5.0f, false);

		cv::Mat thresholdMask;
		cv::threshold(depth, thresholdMask, 2.0, 255, cv::THRESH_BINARY_INV);
		thresholdMask.convertTo(thresholdMask, CV_8U);
		cv::imshow("thresh", thresholdMask);

		cv::Mat masked;
		halfFisheye1.copyTo(masked, ~(thresholdMask)& fisheyeMask8);
		cv::imshow("masked", masked);

		std::string appender;
		if (k < 10) appender = "000";
		else if ((k >= 10) && (k < 100)) appender = "00";
		else if ((k >= 100) && (k < 1000)) appender = "0";
		else appender = "";
		cv::imwrite(mainfolder + outputfolder + "/im" + appender + std::to_string(k) + ".png", masked);

		std::cout << k << std::endl;
		cv::imshow("test", halfFisheye1);
		cv::waitKey(1);
	}
	return 0;
}

int test_VehicleSegmentation() {
	cv::Mat im1 = cv::imread("h:/data_rs_iis/20190913_1/colored_0/data/im1014.png", cv::IMREAD_GRAYSCALE);
	cv::Mat im2 = cv::imread("h:/data_rs_iis/20190913_1/colored_1/data/im1014.png", cv::IMREAD_GRAYSCALE);

	StereoLite * stereotgv = new StereoLite();
	int width = 848;
	int height = 800;
	float stereoScaling = 2.0f;
	int nLevel = 5;
	float fScale = 2.0f;
	int nWarpIters = 10;
	int nSolverIters = 5;
	float lambda = 1.0f;
	float theta = 0.33f;
	float tau = 0.125f;
	stereotgv->limitRange = 0.2f;
	stereotgv->planeSweepMaxError = 1000.0f;
	stereotgv->planeSweepMaxDisparity = (int)(50.0f / stereoScaling);
	stereotgv->planeSweepStride = 1;
	stereotgv->planeSweepWindow = 9;
	stereotgv->planeSweepEpsilon = 1.0f;
	stereotgv->l2lambda = 1.0f;

	int stereoWidth = (int)((float)width / stereoScaling);
	int stereoHeight = (int)((float)height / stereoScaling);
	stereotgv->baseline = 0.0642f;
	stereotgv->focal = 285.8557f / stereoScaling;
	cv::Mat translationVector, calibrationVector;
	if (stereoScaling == 2.0f) {
		translationVector = cv::readOpticalFlow("../test_rstracking/translationVectorHalf.flo");
		calibrationVector = cv::readOpticalFlow("../test_rstracking/calibrationVectorHalf.flo");
	}
	else {
		translationVector = cv::readOpticalFlow("../test_rstracking/translationVector.flo");
		calibrationVector = cv::readOpticalFlow("../test_rstracking/calibrationVector.flo");
	}

	stereotgv->initialize(stereoWidth, stereoHeight, lambda, theta, tau, nLevel, fScale, nWarpIters, nSolverIters);

	// Load fisheye Mask
	cv::Mat fisheyeMask8, fisheyeMask;
	if (stereoScaling == 2.0f) fisheyeMask8 = cv::imread("maskHalf.png", cv::IMREAD_GRAYSCALE);
	else fisheyeMask8 = cv::imread("mask.png", cv::IMREAD_GRAYSCALE);
	fisheyeMask8.convertTo(fisheyeMask, CV_32F, 1.0 / 255.0);
	stereotgv->copyMaskToDevice(fisheyeMask);

	// Load vector fields
	stereotgv->loadVectorFields(translationVector, calibrationVector);

	// Solve stereo depth
	cv::Mat halfFisheye1, halfFisheye2;
	cv::Mat equi1, equi2;
	cv::equalizeHist(im1, equi1);
	cv::equalizeHist(im2, equi2);
	cv::resize(equi1, halfFisheye1, cv::Size(stereoWidth, stereoHeight));
	cv::resize(equi2, halfFisheye2, cv::Size(stereoWidth, stereoHeight));

	cv::Mat depth = cv::Mat(stereoHeight, stereoWidth, CV_32F);
	cv::Mat depthProp, depthTV;
	clock_t start = clock();
	stereotgv->copyImagesToDevice(halfFisheye1, halfFisheye2);
	stereotgv->planeSweepPyramidL1Refinement();
	stereotgv->copyStereoToHost(depth);

	clock_t timeElapsed = (clock() - start);
	std::cout << "time: " << timeElapsed << " ms" << std::endl;

	cv::Mat depthVis, depthVisProp, depthVisTV;
	depth.copyTo(depthVis, fisheyeMask8);
	showDepthJet("color", depthVis, 5.0f, false);

	cv::Mat thresholdMask;
	cv::threshold(depth, thresholdMask, 2.0, 255, cv::THRESH_BINARY_INV);
	thresholdMask.convertTo(thresholdMask, CV_8U);
	cv::imshow("thresh", thresholdMask);

	cv::Mat masked;
	halfFisheye1.copyTo(masked, ~(thresholdMask) & fisheyeMask8);
	cv::imshow("masked", masked);

	cv::waitKey();
	return 0;
}

int test_PlaneSweepWithTvl1() {
	cv::Mat im1 = cv::imread("h:/data_rs_iis/20190913_1/colored_0/data/im1014.png", cv::IMREAD_GRAYSCALE);
	cv::Mat im2 = cv::imread("h:/data_rs_iis/20190913_1/colored_1/data/im1014.png", cv::IMREAD_GRAYSCALE);

	StereoLite * stereotgv = new StereoLite();
	int width = 848;
	int height = 800;
	float stereoScaling = 1.0f;
	int nLevel = 5;
	float fScale = 2.0f;
	int nWarpIters = 10;
	int nSolverIters = 5;
	float lambda = 1.0f;
	float theta = 0.33f;
	float tau = 0.125f;
	stereotgv->limitRange = 0.2f;
	stereotgv->planeSweepMaxError = 1000.0f;
	stereotgv->planeSweepMaxDisparity = (int)(50.0f / stereoScaling);
	stereotgv->planeSweepStride = 1;
	stereotgv->planeSweepWindow = 9;
	stereotgv->planeSweepEpsilon = 1.0f;
	int upsamplingRadius = 5;
	stereotgv->maxPropIter = 3;
	stereotgv->l2lambda = 1.0f;
	stereotgv->l2iter = 500;
	stereotgv->planeSweepStandardDev = 1.0f;

	int stereoWidth = (int)((float)width / stereoScaling);
	int stereoHeight = (int)((float)height / stereoScaling);
	stereotgv->baseline = 0.0642f;
	stereotgv->focal = 285.8557f / stereoScaling;
	cv::Mat translationVector, calibrationVector;
	if (stereoScaling == 2.0f) {
		translationVector = cv::readOpticalFlow("../test_rstracking/translationVectorHalf.flo");
		calibrationVector = cv::readOpticalFlow("../test_rstracking/calibrationVectorHalf.flo");
	}
	else {
		translationVector = cv::readOpticalFlow("../test_rstracking/translationVector.flo");
		calibrationVector = cv::readOpticalFlow("../test_rstracking/calibrationVector.flo");
	}

	stereotgv->initialize(stereoWidth, stereoHeight, lambda, theta, tau, nLevel, fScale, nWarpIters, nSolverIters);

	// Load fisheye Mask
	cv::Mat fisheyeMask8, fisheyeMask;
	if (stereoScaling == 2.0f) fisheyeMask8 = cv::imread("maskHalf.png", cv::IMREAD_GRAYSCALE);
	else fisheyeMask8 = cv::imread("mask.png", cv::IMREAD_GRAYSCALE);
	fisheyeMask8.convertTo(fisheyeMask, CV_32F, 1.0 / 255.0);
	stereotgv->copyMaskToDevice(fisheyeMask);

	// Load vector fields
	stereotgv->loadVectorFields(translationVector, calibrationVector);

	// Solve stereo depth
	cv::Mat halfFisheye1, halfFisheye2;
	cv::Mat equi1, equi2;
	cv::equalizeHist(im1, equi1);
	cv::equalizeHist(im2, equi2);
	cv::resize(equi1, halfFisheye1, cv::Size(stereoWidth, stereoHeight));
	cv::resize(equi2, halfFisheye2, cv::Size(stereoWidth, stereoHeight));

	cv::Mat depth = cv::Mat(stereoHeight, stereoWidth, CV_32F);
	cv::Mat depthProp, depthTV;
	clock_t start = clock();
	stereotgv->copyImagesToDevice(halfFisheye1, halfFisheye2);
	//stereotgv->planeSweepPropagationL1Refinement(upsamplingRadius);
	//stereotgv->copyStereoToHost(depth);
	//stereotgv->planeSweepL1upsamplingAndRefinement();
	//stereotgv->planeSweepL2upsamplingL1Refinement();
	stereotgv->planeSweepPyramidL1Refinement();
	//stereotgv->copyPlanesweepFinalToHost(depth);
	//stereotgv->solveStereoForwardMasked();
	//depthProp = depth.clone();
	stereotgv->copyStereoToHost(depth);
	//stereotgv->planeSweepWithPropagation(upsamplingRadius);
	//stereotgv->copyPropagatedDisparityToHost(depth);
	//depthProp = depth.clone();
	//stereotgv->copyStereoToHost(depth);
	
	clock_t timeElapsed = (clock() - start);
	std::cout << "time: " << timeElapsed << " ms" << std::endl;

	cv::Mat depthVis, depthVisProp, depthVisTV;
	//depthTV.copyTo(depthVisTV, fisheyeMask8);
	//showDepthJet("colorTV", depthVisTV, 5.0f, false);
	//depthProp.copyTo(depthVisProp, fisheyeMask8);
	//showDepthJet("colorProp", depthVisProp, 5.0f, false);
	depth.copyTo(depthVis, fisheyeMask8);
	showDepthJet("color", depthVis, 5.0f, false);

	cv::Mat thresholdMask;
	cv::threshold(depth, thresholdMask, 2.0, 255, cv::THRESH_BINARY_INV);
	thresholdMask.convertTo(thresholdMask, CV_8U);
	cv::imshow("thresh", thresholdMask);

	//cv::imshow("equi1", equi1);
	//cv::imshow("equi2", equi2);
	/*saveDepthJet("resultPS" + std::to_string(nWarpIters) + std::to_string(nSolverIters)
		+ std::to_string(lambda) + std::to_string(timeElapsed) + ".png", depthVis, 5.0f);*/
	cv::waitKey();
	return 0;
}

int test_ImageSequencePlanesweep(std::string mainfolder, int startFrame, int endFrame) {
	//std::string mainfolder = "h:/data_rs_iis/20190913_1";
	//std::string outputfolder = "/pshalf_cons3win9iter5";
	StereoLite * stereotgv = new StereoLite();	
	int width = 848;
	int height = 800;
	float stereoScaling = 2.0f;
	int nLevel = 5;
	float fScale = 2.0f;
	int nWarpIters = 10;
	int nSolverIters = 5;
	float lambda = 1.0f;
	float theta = 0.33f;
	float tau = 0.125f;
	stereotgv->limitRange = 0.2f;
	stereotgv->planeSweepMaxError = 1000.0f;
	stereotgv->planeSweepMaxDisparity = (int)(50.0f);
	stereotgv->planeSweepStride = 0.05f;
	stereotgv->planeSweepWindow = 11;
	stereotgv->planeSweepEpsilon = 0.5f;
	int upsamplingRadius = 5;
	stereotgv->maxPropIter = 3;
	stereotgv->l2lambda = 1.0f;
	stereotgv->l2iter = 500;
	stereotgv->planeSweepStandardDev = 1.0f;
	float maxDepthVis = 20.0f;

	int stereoWidth = (int)((float)width / stereoScaling);
	int stereoHeight = (int)((float)height / stereoScaling);
	stereotgv->baseline = 0.0642f;
	stereotgv->focal = 285.8557f / stereoScaling;

	stereotgv->initialize(stereoWidth, stereoHeight, lambda, theta, tau, nLevel, fScale, nWarpIters, nSolverIters);

	// Load fisheye Mask
	cv::Mat mask = cv::Mat::zeros(cv::Size(stereoWidth, stereoHeight), CV_8UC1);
	circle(mask, cv::Point(stereoWidth / 2, stereoHeight / 2), stereoWidth / 2 - (int)(40 / stereoScaling), cv::Scalar(256.0f), -1);
	cv::Mat fisheyeMask;
	mask.convertTo(fisheyeMask, CV_32F, 1.0 / 255.0);
	//cv::Mat fisheyeMask8, fisheyeMask;
	//if (stereoScaling == 2.0f) fisheyeMask8 = cv::imread("maskHalf.png", cv::IMREAD_GRAYSCALE);
	//else fisheyeMask8 = cv::imread("mask.png", cv::IMREAD_GRAYSCALE);
	//fisheyeMask8.convertTo(fisheyeMask, CV_32F, 1.0 / 255.0);
	//stereotgv->copyMaskToDevice(fisheyeMask);

	// Load vector fields
	cv::Mat translationVector, calibrationVector;
	if (stereoScaling == 2.0f) {
		translationVector = cv::readOpticalFlow("h:/data_rs_iis/translationVectorHalf.flo");
		calibrationVector = cv::readOpticalFlow("h:/data_rs_iis/calibrationVectorHalf.flo");
	}
	else {
		translationVector = cv::readOpticalFlow("h:/data_rs_iis/translationVector.flo");
		calibrationVector = cv::readOpticalFlow("h:/data_rs_iis/calibrationVector.flo");
	}
	stereotgv->loadVectorFields(translationVector, calibrationVector);

	cv::Mat halfFisheye1, halfFisheye2;
	cv::Mat equi1, equi2;

	clock_t avetime = 0;

	for (int k = startFrame; k <= endFrame; k++) {
		cv::Mat im1 = cv::imread(mainfolder + "/colored_0/data/im" + std::to_string(k) + ".png", cv::IMREAD_GRAYSCALE);
		cv::Mat im2 = cv::imread(mainfolder + "/colored_1/data/im" + std::to_string(k) + ".png", cv::IMREAD_GRAYSCALE);

		// Solve stereo depth
		cv::equalizeHist(im1, equi1);
		cv::equalizeHist(im2, equi2);
		cv::resize(equi1, halfFisheye1, cv::Size(stereoWidth, stereoHeight));
		cv::resize(equi2, halfFisheye2, cv::Size(stereoWidth, stereoHeight));

		clock_t start = clock();
		stereotgv->copyImagesToDevice(halfFisheye1, halfFisheye2);
		//stereotgv->planeSweep();
		//stereotgv->planeSweepPropagationL1Refinement(upsamplingRadius);
		//stereotgv->planeSweepWithPropagation(upsamplingRadius);
		//stereotgv->copyPlanesweepFinalToHost(depth);
		//stereotgv->planeSweepL1upsamplingAndRefinement();
		//stereotgv->planeSweepPyramidL1Refinement();
		//stereotgv->solveStereoForwardMasked();
		//stereotgv->copyPlanesweepFinalToHost(depth);

		stereotgv->planeSweepSubpixel();
		
		cv::Mat depth = cv::Mat(stereoHeight, stereoWidth, CV_32F);
		//stereotgv->copyStereoToHost(depth);
		//stereotgv->copyPlanesweepFinalToHost(depth);
		stereotgv->copyPlanesweepForwardToHost(depth);

		//cv::Mat depthPs = cv::Mat(stereoHeight, stereoWidth, CV_32F);
		//depthPs = depth.clone();

		//stereotgv->copyPropagatedDisparityToHost(depth);
		clock_t timeElapsed = (clock() - start);
		//avetime = (avetime + timeElapsed) / 2;
		std::cout << "time: " << timeElapsed << " ms ";

		cv::Mat depthVis;
		depth.copyTo(depthVis, mask);
		showDepthJetExponential("color", depthVis, maxDepthVis, 0.1f, false);
		
		/*cv::Mat depthVisPs;
		depthPs.copyTo(depthVisPs, fisheyeMask8);
		showDepthJet("ps", depthVisPs, 5.0f, false);*/

		std::string appender;
		if (k < 10) appender = "000";
		else if ((k >= 10) && (k < 100)) appender = "00";
		else if ((k >= 100) && (k < 1000)) appender = "0";
		else appender = "";

		cv::Mat depth16;
		depthVis.convertTo(depth16, CV_16U, 256.0f);
		cv::imwrite(mainfolder + "/outputplanesweep/im" + appender + std::to_string(k) + ".png", depth16);
		saveDepthJetExponential(mainfolder + "/error/planesweep/im" + appender + std::to_string(k) + ".png", depthVis, maxDepthVis, 0.1f);
		//saveDepthJet(mainfolder + outputfolder + "/im" + appender + std::to_string(k) + ".png", depthVis, 5.0f);

		std::cout << k << std::endl;
		cv::imshow("test", halfFisheye1);
		cv::waitKey(1);
	}
	return 0;
}

int test_PlaneSweep() {
	cv::Mat im1 = cv::imread("h:/data_rs_iis/20190913_1/colored_0/data/im1.png", cv::IMREAD_GRAYSCALE);
	cv::Mat im2 = cv::imread("h:/data_rs_iis/20190913_1/colored_1/data/im1.png", cv::IMREAD_GRAYSCALE);

	StereoLite * stereotgv = new StereoLite();
	int width = 848;
	int height = 800;
	float stereoScaling = 1.0f;
	int nLevel = 5;
	float fScale = 2.0f;
	int nWarpIters = 10;
	int nSolverIters = 20;
	float lambda = 1.5f;
	float theta = 0.33f;
	float tau = 0.125f;
	stereotgv->limitRange = 0.2f;
	stereotgv->planeSweepMaxError = 1000.0f;
	stereotgv->planeSweepMaxDisparity = 50;
	stereotgv->planeSweepStride = 1;
	stereotgv->planeSweepWindow = 5;
	stereotgv->planeSweepEpsilon = 1.0f;

	int stereoWidth = (int)((float)width / stereoScaling);
	int stereoHeight = (int)((float)height / stereoScaling);
	stereotgv->baseline = 0.0642f;
	stereotgv->focal = 285.8557f / stereoScaling;
	cv::Mat translationVector, calibrationVector;
	if (stereoScaling == 2.0f) {
		translationVector = cv::readOpticalFlow("../test_rstracking/translationVectorHalf.flo");
		calibrationVector = cv::readOpticalFlow("../test_rstracking/calibrationVectorHalf.flo");
	}
	else {
		translationVector = cv::readOpticalFlow("../test_rstracking/translationVector.flo");
		calibrationVector = cv::readOpticalFlow("../test_rstracking/calibrationVector.flo");
	}

	stereotgv->initialize(stereoWidth, stereoHeight, lambda, theta, tau, nLevel, fScale, nWarpIters, nSolverIters);

	// Load fisheye Mask
	cv::Mat fisheyeMask8;
	if (stereoScaling == 2.0f) {
		fisheyeMask8 = cv::imread("maskHalf.png", cv::IMREAD_GRAYSCALE);
	}
	else {
		fisheyeMask8 = cv::imread("mask.png", cv::IMREAD_GRAYSCALE);
	}
	cv::Mat fisheyeMask;
	fisheyeMask8.convertTo(fisheyeMask, CV_32F, 1.0 / 255.0);
	stereotgv->copyMaskToDevice(fisheyeMask);

	// Load vector fields
	stereotgv->loadVectorFields(translationVector, calibrationVector);

	// Solve stereo depth
	cv::Mat halfFisheye1, halfFisheye2;
	cv::Mat equi1, equi2;
	cv::equalizeHist(im1, equi1);
	cv::equalizeHist(im2, equi2);
	cv::resize(equi1, halfFisheye1, cv::Size(stereoWidth, stereoHeight));
	cv::resize(equi2, halfFisheye2, cv::Size(stereoWidth, stereoHeight));

	cv::Mat depth = cv::Mat(stereoHeight, stereoWidth, CV_32F);
	cv::Mat depthF = cv::Mat(stereoHeight, stereoWidth, CV_32F);
	cv::Mat depthTV;
	clock_t start = clock();
	stereotgv->copyImagesToDevice(halfFisheye1, halfFisheye2);
	stereotgv->planeSweep();
	//stereotgv->copyPlanesweepForwardToHost(depth);
	//depthF = depth.clone();
	//stereotgv->copyPlanesweepBackwardToHost(depthB);
	stereotgv->copyPlanesweepFinalToHost(depth);
	clock_t timeElapsed = (clock() - start);
	std::cout << "time: " << timeElapsed << " ms" << std::endl;

	cv::Mat depthVis, depthVisTV;
	//depthTV.copyTo(depthVisTV, fisheyeMask8);
	//showDepthJet("colorTV", depthVisTV, 5.0f, false);
	depth.copyTo(depthVis, fisheyeMask8);
	showDepthJet("color", depthVis, 5.0f, false);
	//cv::imshow("equi1", equi1);
	//cv::imshow("equi2", equi2);
	/*saveDepthJet("resultPS" + std::to_string(nWarpIters) + std::to_string(nSolverIters)
		+ std::to_string(lambda) + std::to_string(timeElapsed) + ".png", depthVis, 5.0f);*/
	cv::waitKey();
	return 0;
}

int test_StereoLiteTwoFrames(int level, float scale , int wapriters, int solveriters) {
	cv::Mat im1 = cv::imread("h:/data_rs_iis/20190913_1/colored_0/data/im174.png", cv::IMREAD_GRAYSCALE);
	cv::Mat im2 = cv::imread("h:/data_rs_iis/20190913_1/colored_1/data/im174.png", cv::IMREAD_GRAYSCALE);

	StereoLite * stereotgv = new StereoLite();
	int width = 848;
	int height = 800;
	float stereoScaling = 2.0f;
	int nLevel = 4;
	float fScale = 2.0f;
	int nWarpIters = 10;
	int nSolverIters = 20;
	float lambda = 1.5f;
	float theta = 0.33f;
	float tau = 0.125f;
	stereotgv->limitRange = 0.2f;

	int stereoWidth = (int)((float)width / stereoScaling);
	int stereoHeight = (int)((float)height / stereoScaling);
	stereotgv->baseline = 0.0642f;
	stereotgv->focal = 285.8557f / stereoScaling;
	cv::Mat translationVector, calibrationVector;
	if (stereoScaling == 2.0f) {
		translationVector = cv::readOpticalFlow("../test_rstracking/translationVectorHalf.flo");
		calibrationVector = cv::readOpticalFlow("../test_rstracking/calibrationVectorHalf.flo");
	}
	else {
		translationVector = cv::readOpticalFlow("../test_rstracking/translationVector.flo");
		calibrationVector = cv::readOpticalFlow("../test_rstracking/calibrationVector.flo");
	}

	stereotgv->initialize(stereoWidth, stereoHeight, lambda, theta, tau, nLevel, fScale, nWarpIters, nSolverIters);

	// Load fisheye Mask
	cv::Mat fisheyeMask8;
	if (stereoScaling == 2.0f) {
		fisheyeMask8 = cv::imread("maskHalf.png", cv::IMREAD_GRAYSCALE);
	}
	else {
		fisheyeMask8 = cv::imread("mask.png", cv::IMREAD_GRAYSCALE);
	}
	cv::Mat fisheyeMask;
	fisheyeMask8.convertTo(fisheyeMask, CV_32F, 1.0 / 255.0);
	//cv::imshow("fm", fisheyeMask);
	stereotgv->copyMaskToDevice(fisheyeMask);

	// Load vector fields
	stereotgv->loadVectorFields(translationVector, calibrationVector);

	// Solve stereo depth
	cv::Mat halfFisheye1, halfFisheye2;
	cv::Mat equi1, equi2;
	cv::equalizeHist(im1, equi1);
	cv::equalizeHist(im2, equi2);
	cv::resize(equi1, halfFisheye1, cv::Size(stereoWidth, stereoHeight));
	cv::resize(equi2, halfFisheye2, cv::Size(stereoWidth, stereoHeight));

	cv::Mat depth = cv::Mat(stereoHeight, stereoWidth, CV_32F);
	clock_t start = clock();
	stereotgv->copyImagesToDevice(halfFisheye1, halfFisheye2);
	stereotgv->solveStereoForwardMasked();
	stereotgv->copyStereoToHost(depth);
	clock_t timeElapsed = (clock() - start);
	std::cout << "time: " << timeElapsed << " ms" << std::endl;

	cv::Mat depthVis;
	depth.copyTo(depthVis, fisheyeMask8);
	showDepthJet("color", depthVis, 3.0f, false);
	saveDepthJet("resultTVL1" + std::to_string(nWarpIters) + std::to_string(nSolverIters) 
		+ std::to_string(lambda) + std::to_string(timeElapsed) + ".png", depthVis, 3.0f);
	cv::waitKey();
	return 0;
}

int test_TwoFrames(int nLevel, float fScale, int nWarpIters, int nSolverIters) {
	cv::Mat im1 = cv::imread("h:/data_rs_iis/20190913_1/colored_0/data/im174.png", cv::IMREAD_GRAYSCALE);
	cv::Mat im2 = cv::imread("h:/data_rs_iis/20190913_1/colored_1/data/im174.png", cv::IMREAD_GRAYSCALE);

	StereoTgv * stereotgv = new StereoTgv();
	int width = 848;
	int height = 800;
	float stereoScaling = 1.0f;
	/*int nLevel = nLevel;
	float fScale = fScale;
	int nWarpIters = nWarpIters;
	int nSolverIters = nSolverIters;*/
	float lambda = 5.0f;

	int stereoWidth = (int)((float)width / stereoScaling);
	int stereoHeight = (int)((float)height / stereoScaling);
	stereotgv->baseline = 0.0642f;
	stereotgv->focal = 285.8557f / stereoScaling;
	cv::Mat translationVector, calibrationVector;
	if (stereoScaling == 2.0f) {
		translationVector = cv::readOpticalFlow("../test_rstracking/translationVectorHalf.flo");
		calibrationVector = cv::readOpticalFlow("../test_rstracking/calibrationVectorHalf.flo");
	}
	else {
		translationVector = cv::readOpticalFlow("../test_rstracking/translationVector.flo");
		calibrationVector = cv::readOpticalFlow("../test_rstracking/calibrationVector.flo");
	}

	float beta = 4.0f;
	float gamma = 0.2f;
	float alpha0 = 17.0f;
	float alpha1 = 1.2f;
	float timeStepLambda = 1.0f;

	stereotgv->initialize(stereoWidth, stereoHeight, beta, gamma, alpha0, alpha1,
		timeStepLambda, lambda, nLevel, fScale, nWarpIters, nSolverIters);
	stereotgv->visualizeResults = true;

	// Load fisheye Mask
	cv::Mat fisheyeMask8;
	if (stereoScaling == 2.0f) {
		fisheyeMask8 = cv::imread("maskHalf.png", cv::IMREAD_GRAYSCALE);
	}
	else {
		fisheyeMask8 = cv::imread("mask.png", cv::IMREAD_GRAYSCALE);
	}
	cv::Mat fisheyeMask;
	fisheyeMask8.convertTo(fisheyeMask, CV_32F, 1.0 / 255.0);
	//cv::imshow("fm", fisheyeMask);
	stereotgv->copyMaskToDevice(fisheyeMask);

	// Load vector fields
	stereotgv->loadVectorFields(translationVector, calibrationVector);

	// Solve stereo depth
	cv::Mat halfFisheye1, halfFisheye2;
	cv::Mat equi1, equi2;
	cv::equalizeHist(im1, equi1);
	cv::equalizeHist(im2, equi2);
	cv::resize(equi1, halfFisheye1, cv::Size(stereoWidth, stereoHeight));
	cv::resize(equi2, halfFisheye2, cv::Size(stereoWidth, stereoHeight));

	cv::Mat depth = cv::Mat(stereoHeight, stereoWidth, CV_32F);
	clock_t start = clock();
	stereotgv->copyImagesToDevice(halfFisheye1, halfFisheye2);
	stereotgv->solveStereoForwardMasked();
	stereotgv->copyStereoToHost(depth);
	clock_t timeElapsed = (clock() - start);
	std::cout << "time: " << timeElapsed << " ms" << std::endl;

	cv::Mat depthVis;
	depth.copyTo(depthVis, fisheyeMask8);
	showDepthJet("color2", depthVis, 30.0f, false);
	saveDepthJet("resultTGVL1.png", depthVis, 30.0f);
	cv::waitKey();
	return 0;
}

int test_BlenderDataSequence() {
	std::string folder = "D:/dev/blender/icra2020";
	cv::Mat translationVector = cv::readOpticalFlow(folder + "/translationVectorBlender.flo");
	int stereoWidth = 800;
	int stereoHeight = 800;
	cv::Mat calibrationVector = cv::Mat::zeros(cv::Size(stereoWidth, stereoHeight), CV_32FC2);
	cv::Mat mask = cv::Mat::zeros(cv::Size(stereoWidth, stereoHeight), CV_8UC1);
	circle(mask, cv::Point(stereoWidth / 2, stereoHeight / 2), stereoWidth / 2 - 50, cv::Scalar(256.0f), -1);
	cv::Mat fisheyeMask;
	mask.convertTo(fisheyeMask, CV_32F, 1.0 / 255.0);

	StereoTgv * stereotgv = new StereoTgv();
	int width = 800;
	int height = 800;
	float stereoScaling = 1.0f;
	int nLevel = 11; // 11 in paper
	float fScale = 1.2f;
	int nWarpIters = 50;
	int nSolverIters = 50;
	float lambda = 5.0f;
	stereotgv->limitRange = 0.1f;


	float beta = 9.0f;
	float gamma = 0.85f;
	float alpha0 = 17.0f;
	float alpha1 = 1.2f;
	float timeStepLambda = 1.0f;

	stereotgv->initialize(stereoWidth, stereoHeight, beta, gamma, alpha0, alpha1,
		timeStepLambda, lambda, nLevel, fScale, nWarpIters, nSolverIters);
	stereotgv->visualizeResults = true;

	stereotgv->copyMaskToDevice(fisheyeMask);
	stereotgv->loadVectorFields(translationVector, calibrationVector);
	cv::Mat equi1, equi2;

	for (int k = 1; k <= 300; k++) {
		std::string appender;
		if (k < 10) appender = "000";
		else if ((k >= 10) && (k < 100)) appender = "00";
		else if ((k >= 100) && (k < 1000)) appender = "0";
		else appender = "";
		std::string outputFilename = folder + "/videoflow/" + appender + std::to_string(k) + ".flo";
		cv::Mat im1 = cv::imread(folder + "/left/" + appender + std::to_string(k) + ".png", cv::IMREAD_GRAYSCALE);
		cv::Mat im2 = cv::imread(folder + "/right/" + appender + std::to_string(k) + ".png", cv::IMREAD_GRAYSCALE);

		cv::equalizeHist(im1, equi1);
		cv::equalizeHist(im2, equi2);
		//stereotgv->copyImagesToDevice(im1, im2);
		stereotgv->copyImagesToDevice(equi1, equi2);
		stereotgv->solveStereoForwardMasked();

		cv::Mat disparity = cv::Mat(stereoHeight, stereoWidth, CV_32FC2);
		stereotgv->copyDisparityToHost(disparity);
		cv::writeOpticalFlow(outputFilename, disparity);

		cv::imshow("left", im1);
		cv::waitKey(1);
		std::cout << k << std::endl;
	}
}

int test_ImageSequenceLite() {
	std::string mainfolder = "h:/data_rs_iis/20190913_1";
	std::string outputfolder = "/tvl1_20fps_full";
	StereoLite * stereotgv = new StereoLite();
	int width = 848;
	int height = 800;
	float stereoScaling = 1.0f;
	int nLevel = 5;
	float fScale = 2.0f;
	int nWarpIters = 10;
	int nSolverIters = 20;
	float lambda = 1.0f;
	float theta = 0.33f;
	float tau = 0.125f;
	stereotgv->limitRange = 0.2f;

	int stereoWidth = (int)((float)width / stereoScaling);
	int stereoHeight = (int)((float)height / stereoScaling);
	stereotgv->baseline = 0.0642f;
	stereotgv->focal = 285.8557f / stereoScaling;
	cv::Mat translationVector, calibrationVector;
	if (stereoScaling == 2.0f) {
		translationVector = cv::readOpticalFlow("../test_rstracking/translationVectorHalf.flo");
		calibrationVector = cv::readOpticalFlow("../test_rstracking/calibrationVectorHalf.flo");
	}
	else {
		translationVector = cv::readOpticalFlow("../test_rstracking/translationVector.flo");
		calibrationVector = cv::readOpticalFlow("../test_rstracking/calibrationVector.flo");
	}

	stereotgv->initialize(stereoWidth, stereoHeight, lambda, theta, tau, nLevel, fScale, nWarpIters, nSolverIters);

	cv::Mat mask = cv::Mat::zeros(cv::Size(stereoWidth, stereoHeight), CV_8UC1);
	circle(mask, cv::Point(stereoWidth / 2, stereoHeight / 2), stereoWidth / 2 - 40, cv::Scalar(256.0f), -1);
	cv::Mat fisheyeMask;
	mask.convertTo(fisheyeMask, CV_32F, 1.0 / 255.0);

	stereotgv->copyMaskToDevice(fisheyeMask);

	// Load vector fields
	stereotgv->loadVectorFields(translationVector, calibrationVector);

	cv::Mat halfFisheye1, halfFisheye2;
	cv::Mat equi1, equi2;

	clock_t avetime = 0;

	for (int k = 0; k <= 1274; k++) {
		cv::Mat im1 = cv::imread(mainfolder + "/colored_0/data/im" + std::to_string(k) + ".png", cv::IMREAD_GRAYSCALE);
		cv::Mat im2 = cv::imread(mainfolder + "/colored_1/data/im" + std::to_string(k) + ".png", cv::IMREAD_GRAYSCALE);

		// Solve stereo depth
		cv::equalizeHist(im1, equi1);
		cv::equalizeHist(im2, equi2);

		cv::resize(equi1, halfFisheye1, cv::Size(stereoWidth, stereoHeight));
		cv::resize(equi2, halfFisheye2, cv::Size(stereoWidth, stereoHeight));
		cv::Mat half1, half2;
		cv::resize(im1, half1, cv::Size(stereoWidth, stereoHeight));
		cv::resize(im2, half2, cv::Size(stereoWidth, stereoHeight));

		clock_t start = clock();
		stereotgv->copyImagesToDevice(halfFisheye1, halfFisheye2);
		stereotgv->solveStereoForwardMasked();
		cv::Mat depth = cv::Mat(stereoHeight, stereoWidth, CV_32F);
		cv::Mat depthVis;
		stereotgv->copyStereoToHost(depth);
		clock_t timeElapsed = (clock() - start);
		avetime = (avetime + timeElapsed) / 2;
		std::cout << "time: " << avetime << " ms ";
		//stereotgv->copy1DDisparityToHost(depth);
		depth.copyTo(depthVis, mask);
		showDepthJet("color", depthVis, 3.0f, false);

		std::string appender;
		if (k < 10) appender = "000";
		else if ((k >= 10) && (k < 100)) appender = "00";
		else if ((k >= 100) && (k < 1000)) appender = "0";
		else appender = "";
		saveDepthJet(mainfolder + outputfolder + "/im" + appender + std::to_string(k) + ".png", depthVis, 5.0f);

		std::cout << k << std::endl;
		cv::imshow("test", halfFisheye1);
		cv::waitKey(1);
	}
	return 0;
}

int test_ImageSequenceFlow(std::string mainfolder, int startFrame, int endFrame, float scaling) {
	//std::string mainfolder = "h:/data_rs_iis/20191108_20";
	StereoTgv * stereotgv = new StereoTgv();
	int width = 848;
	int height = 800;
	//float stereoScaling = 1.0f;
	float stereoScaling = scaling;
	int nLevel = 11;
	float fScale = 1.2f;// 2.0f;
	int nWarpIters = 400;// 10;
	int nSolverIters = 20;// 20;
	float lambda = 5.0f;
	stereotgv->limitRange = 0.1f;
	float maxDepthVis = 20.0f;
	double sigmaBlur = 0.1;

	int stereoWidth = (int)((float)width / stereoScaling);
	int stereoHeight = (int)((float)height / stereoScaling);
	stereotgv->baseline = 0.0642f;
	stereotgv->focal = 285.8557f / stereoScaling;
	cv::Mat translationVector, calibrationVector;
	if (stereoScaling == 2.0f) {
		translationVector = cv::readOpticalFlow("h:/data_rs_iis/translationVectorHalf.flo");
		calibrationVector = cv::readOpticalFlow("h:/data_rs_iis/calibrationVectorHalf.flo");
	}
	else {
		translationVector = cv::readOpticalFlow("h:/data_rs_iis/translationVector.flo");
		calibrationVector = cv::readOpticalFlow("h:/data_rs_iis/calibrationVector.flo");
	}

	float beta = 4.0f;
	float gamma = 0.2f;
	float alpha0 = 17.0f;
	float alpha1 = 1.2f;
	float timeStepLambda = 1.0f;

	stereotgv->initialize(stereoWidth, stereoHeight, beta, gamma, alpha0, alpha1,
		timeStepLambda, lambda, nLevel, fScale, nWarpIters, nSolverIters);
	stereotgv->visualizeResults = true;

	cv::Mat mask = cv::Mat::zeros(cv::Size(stereoWidth, stereoHeight), CV_8UC1);
	circle(mask, cv::Point(stereoWidth / 2, stereoHeight / 2), stereoWidth / 2 - (int)(40 / stereoScaling), cv::Scalar(256.0f), -1);
	cv::Mat fisheyeMask;
	mask.convertTo(fisheyeMask, CV_32F, 1.0 / 255.0);

	stereotgv->copyMaskToDevice(fisheyeMask);

	// Load vector fields
	stereotgv->loadVectorFields(translationVector, calibrationVector);

	cv::Mat halfFisheye1, halfFisheye2;
	cv::Mat equi1, equi2;

	for (int k = startFrame; k <= endFrame; k++) {
		std::string appender;
		if (k < 10) appender = "000";
		else if ((k >= 10) && (k < 100)) appender = "00";
		else if ((k >= 100) && (k < 1000)) appender = "0";
		else appender = "";

		std::string filename = "im" + std::to_string(k);
		std::string outputFilename = mainfolder + "/output/" + "im" + appender + std::to_string(k) + ".flo";
		std::string outputVisFilename = mainfolder + "/output/vis/" + "im" + appender + std::to_string(k) + ".png";
		std::cout << outputVisFilename << std::endl;
		cv::Mat im1 = cv::imread(mainfolder + "/colored_0/data/" + filename + ".png", cv::IMREAD_GRAYSCALE);
		cv::Mat im2 = cv::imread(mainfolder + "/colored_1/data/" + filename + ".png", cv::IMREAD_GRAYSCALE);

		// Solve stereo depth
		cv::equalizeHist(im1, equi1);
		cv::equalizeHist(im2, equi2);

		cv::resize(equi1, halfFisheye1, cv::Size(stereoWidth, stereoHeight));
		cv::resize(equi2, halfFisheye2, cv::Size(stereoWidth, stereoHeight));

		cv::Mat disparityVis = cv::Mat(stereoHeight, stereoWidth, CV_32FC3);
		cv::Mat disparity = cv::Mat(stereoHeight, stereoWidth, CV_32FC2);

		clock_t start = clock();

		stereotgv->copyImagesToDevice(halfFisheye1, halfFisheye2);
		stereotgv->solveStereoForwardMasked();
		stereotgv->copyDisparityVisToHost(disparityVis, 50.0f);
		stereotgv->copyDisparityToHost(disparity);

		clock_t timeElapsed = (clock() - start);
		std::cout << "time: " << timeElapsed << " ms" << std::endl;

		cv::imshow("flow", disparityVis);
		cv::Mat disparityVis8;
		disparityVis.convertTo(disparityVis8, CV_8UC3, 255.0);
		cv::imwrite(outputVisFilename, disparityVis8);
		cv::writeOpticalFlow(outputFilename, disparity);

		std::cout << k << std::endl;
		cv::waitKey(1);
	}
	cv::waitKey();
	return 0;
}

int test_ImageSequence(std::string mainfolder, int startFrame, int endFrame) {
	//std::string mainfolder = "h:/data_rs_iis/20191108_20";
	StereoTgv * stereotgv = new StereoTgv();
	int width = 848;
	int height = 800;
	float stereoScaling = 1.0f;
	int nLevel = 11;
	float fScale = 1.2f;// 2.0f;
	int nWarpIters = 400;// 10;
	int nSolverIters = 20;// 20;
	float lambda = 5.0f;
	stereotgv->limitRange = 0.1f;
	float maxDepthVis = 20.0f;
	double sigmaBlur = 0.1;

	int stereoWidth = (int)((float)width / stereoScaling);
	int stereoHeight = (int)((float)height / stereoScaling);
	stereotgv->baseline = 0.0642f;
	stereotgv->focal = 285.8557f / stereoScaling;
	cv::Mat translationVector, calibrationVector;
	if (stereoScaling == 2.0f) {
		translationVector = cv::readOpticalFlow("h:/data_rs_iis/translationVectorHalf.flo");
		calibrationVector = cv::readOpticalFlow("h:/data_rs_iis/calibrationVectorHalf.flo");
	}
	else {
		translationVector = cv::readOpticalFlow("h:/data_rs_iis/translationVector.flo");
		calibrationVector = cv::readOpticalFlow("h:/data_rs_iis/calibrationVector.flo");
	}

	float beta = 4.0f;
	float gamma = 0.2f;
	float alpha0 = 17.0f;
	float alpha1 = 1.2f;
	float timeStepLambda = 1.0f;

	stereotgv->initialize(stereoWidth, stereoHeight, beta, gamma, alpha0, alpha1,
		timeStepLambda, lambda, nLevel, fScale, nWarpIters, nSolverIters);
	stereotgv->visualizeResults = true;

	//// Load fisheye Mask
	//cv::Mat fisheyeMask8;
	//if (stereoScaling == 2.0f) {
	//	fisheyeMask8 = cv::imread("maskHalf.png", cv::IMREAD_GRAYSCALE);
	//}
	//else {
	//	//fisheyeMask8 = cv::Mat::zeros(cv::Size(stereoWidth, stereoHeight), CV_8UC1);
	//	//circle(fisheyeMask8, cv::Point(stereoWidth / 2, stereoHeight / 2), stereoWidth / 2 - 70, cv::Scalar(256.0f), -1);
	//	fisheyeMask8 = cv::imread("mask.png", cv::IMREAD_GRAYSCALE);
	//}
	//cv::Mat fisheyeMask;
	//fisheyeMask8.convertTo(fisheyeMask, CV_32F, 1.0 / 255.0);
	//cv::imshow("fm", fisheyeMask);
	cv::Mat mask = cv::Mat::zeros(cv::Size(stereoWidth, stereoHeight), CV_8UC1);
	circle(mask, cv::Point(stereoWidth / 2, stereoHeight / 2), stereoWidth / 2 - (int)(40/stereoScaling), cv::Scalar(256.0f), -1);
	cv::Mat fisheyeMask;
	mask.convertTo(fisheyeMask, CV_32F, 1.0 / 255.0);

	stereotgv->copyMaskToDevice(fisheyeMask);

	// Load vector fields
	stereotgv->loadVectorFields(translationVector, calibrationVector);

	cv::Mat halfFisheye1, halfFisheye2;
	cv::Mat equi1, equi2;
	
	for (int k = startFrame; k <= endFrame; k++) {
		cv::Mat im1 = cv::imread(mainfolder + "/colored_0/data/im" + std::to_string(k) + ".png", cv::IMREAD_GRAYSCALE);
		cv::Mat im2 = cv::imread(mainfolder + "/colored_1/data/im" + std::to_string(k) + ".png", cv::IMREAD_GRAYSCALE);
		
		//cv::Mat depthVisMask = cv::Mat::zeros(cv::Size(stereoWidth, stereoHeight), CV_8UC1);
		//circle(depthVisMask, cv::Point(stereoWidth / 2, stereoHeight / 2), (int)((float)stereoWidth / 2.2f), cv::Scalar(256.0f), -1);

		// Solve stereo depth
		cv::equalizeHist(im1, equi1);
		cv::equalizeHist(im2, equi2);

		/*cv::Mat i1fixed = cv::imread("fs1.png", cv::IMREAD_GRAYSCALE);
		cv::Mat i2fixed = cv::imread("fs2.png", cv::IMREAD_GRAYSCALE);
		cv::equalizeHist(i1fixed, equi1);
		cv::equalizeHist(i2fixed, equi2);*/
		//cv::imshow("equi", equi1);
		cv::resize(equi1, halfFisheye1, cv::Size(stereoWidth, stereoHeight));
		cv::resize(equi2, halfFisheye2, cv::Size(stereoWidth, stereoHeight));
		//cv::Mat half1, half2;
		//cv::resize(im1, half1, cv::Size(stereoWidth, stereoHeight));
		//cv::resize(im2, half2, cv::Size(stereoWidth, stereoHeight));
		//cv::resize(t265.fisheye1, halfFisheye1, cv::Size(stereoWidth, stereoHeight));
		//cv::resize(t265.fisheye2, halfFisheye2, cv::Size(stereoWidth, stereoHeight));
		clock_t start = clock();
		//std::cout << (int)halfFisheye1.at<uchar>(200, 200) << std::endl;

		//cv::GaussianBlur(halfFisheye1, halfFisheye1, cv::Size(5, 5), sigmaBlur, 0.0);
		//cv::GaussianBlur(halfFisheye2, halfFisheye2, cv::Size(5, 5), sigmaBlur, 0.0);
		//cv::medianBlur(halfFisheye1, halfFisheye1, 3);
		//cv::medianBlur(halfFisheye2, halfFisheye2, 3);

		stereotgv->copyImagesToDevice(halfFisheye1, halfFisheye2);
		//stereotgv->copyImagesToDevice(half1, half2);

		//stereotgv->solveStereoForward();
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
		cv::imwrite(mainfolder + "/output/im" + appender + std::to_string(k) + ".png", depth16);
		saveDepthJetExponential(mainfolder + "/error/ours/im" + appender + std::to_string(k) + ".png", depthVis, maxDepthVis, 0.1f);
		//saveDepthJet("h:/data_rs_iis/20190909/output1ddisparity/im" + std::to_string(k) + ".png", depthVis, 30.0f);
		/*cv::Mat imout;
		im1.copyTo(imout, fisheyeMask8);
		cv::imwrite("d:/data/20190909/smalloutputimage/im" + appender + std::to_string(k) + ".png", imout);*/
		std::cout << k << std::endl;
		//cv::imshow("test", halfFisheye1);
		cv::waitKey(1);
	}
	cv::waitKey();
	return 0;
}

int test_Timing(int warpIteration) {
	//std::string folder = "C:/Users/cvl-menandro/Downloads/rpg_urban_blender.tar/rpg_urban_blender";
	std::string folder = "D:/dev/blender/icra2020";
	std::string outputFilename = folder + "/output/timing06_" + std::to_string(warpIteration) + ".flo";
	cv::Mat im1 = cv::imread(folder + "/image/left06.png");
	cv::Mat im2 = cv::imread(folder + "/image/right06.png");
	int stereoWidth = im1.cols;
	int stereoHeight = im1.rows;
	cv::Mat translationVector = cv::readOpticalFlow("D:/dev/matlab_house/translationVectorBlender.flo");
	cv::Mat calibrationVector = cv::Mat::zeros(cv::Size(stereoWidth, stereoHeight), CV_32FC2);
	cv::Mat mask = cv::Mat::zeros(cv::Size(stereoWidth, stereoHeight), CV_8UC1);
	circle(mask, cv::Point(stereoWidth / 2, stereoHeight / 2), stereoWidth / 2 - 50, cv::Scalar(256.0f), -1);
	//cv::imwrite("maskBlender.png", mask);
	//return 0;
	/*cv::imshow("mask", mask);
	std::cout << (int)mask.at<unsigned char>(400, 400) << std::endl;
	cv::waitKey();*/
	cv::Mat fisheyeMask;
	mask.convertTo(fisheyeMask, CV_32F, 1.0 / 255.0);

	StereoTgv * stereotgv = new StereoTgv();
	int width = 800;
	int height = 800;
	float stereoScaling = 1.0f;
	int nLevel = 5;
	float fScale = 2.0;
	int nWarpIters = warpIteration;
	int nSolverIters = 10;
	float lambda = 5.0;
	stereotgv->limitRange = 0.2f;


	float beta = 9.0f;
	float gamma = 0.85f;
	float alpha0 = 17.0f;
	float alpha1 = 1.2f;
	float timeStepLambda = 1.0f;

	stereotgv->initialize(stereoWidth, stereoHeight, beta, gamma, alpha0, alpha1,
		timeStepLambda, lambda, nLevel, fScale, nWarpIters, nSolverIters);
	stereotgv->visualizeResults = true;

	stereotgv->copyMaskToDevice(fisheyeMask);
	stereotgv->loadVectorFields(translationVector, calibrationVector);
	stereotgv->copyImagesToDevice(im1, im2);
	clock_t start = clock();
	stereotgv->solveStereoForwardMasked();
	cv::Mat disparityVis = cv::Mat(stereoHeight, stereoWidth, CV_32FC3);
	//stereotgv->copyDisparityVisToHost(disparityVis, 50.0f);
	//cv::imshow("flow", disparityVis);
	cv::Mat disparity = cv::Mat(stereoHeight, stereoWidth, CV_32FC2);
	stereotgv->copyDisparityToHost(disparity);
	clock_t timeElapsed = (clock() - start);
	std::cout << "time: " << timeElapsed << " ms" << std::endl;
	cv::writeOpticalFlow(outputFilename, disparity);
	// convert disparity to 3D (depends on the model)

	cv::Mat depthVis;
	cv::Mat depth = cv::Mat(stereoHeight, stereoWidth, CV_32F);
	stereotgv->copyStereoToHost(depth);
	depth.copyTo(depthVis, mask);
	//showDepthJet("color", depthVis, 5.0f, false);

	cv::Mat warped;
	stereotgv->copyWarpedImageToHost(warped);
	//cv::imshow("left", im1);
	//cv::imshow("right", im2);
	//cv::imshow("warped", warped);
	//cv::waitKey();
}

int test_LimitingRangeOne() {
	//std::string folder = "C:/Users/cvl-menandro/Downloads/rpg_urban_blender.tar/rpg_urban_blender";
	std::string folder = "D:/dev/blender/icra2020";
	cv::Mat im1 = cv::imread(folder + "/image/left06.png", cv::IMREAD_GRAYSCALE);
	cv::Mat im2 = cv::imread(folder + "/image/right06.png", cv::IMREAD_GRAYSCALE);
	int stereoWidth = im1.cols;
	int stereoHeight = im1.rows;
	cv::Mat translationVector = cv::readOpticalFlow("D:/dev/matlab_house/translationVectorBlender.flo");
	cv::Mat calibrationVector = cv::Mat::zeros(cv::Size(stereoWidth, stereoHeight), CV_32FC2);
	cv::Mat mask = cv::Mat::zeros(cv::Size(stereoWidth, stereoHeight), CV_8UC1);
	circle(mask, cv::Point(stereoWidth / 2, stereoHeight / 2), stereoWidth / 2 - 50, cv::Scalar(256.0f), -1);
	cv::Mat fisheyeMask;
	mask.convertTo(fisheyeMask, CV_32F, 1.0 / 255.0);

	
	int width = 800;
	int height = 800;
	float stereoScaling = 1.0f;
	int nLevel = 11;
	float fScale = 1.2;
	
	int nSolverIters = 50;
	float lambda = 5.0;
	float beta = 9.0f;
	float gamma = 0.85f;
	float alpha0 = 17.0f;
	float alpha1 = 1.2f;
	float timeStepLambda = 1.0f;

	int nWarpIters = 100;

	for (int range = 1; range <= 10; range++) {
		StereoTgv * stereotgv = new StereoTgv();
		stereotgv->limitRange = 0.1f * (float)range;
		std::string appender;
		if (range < 10) appender = "0";
		else appender = "";
		std::string outputFilename = folder + "/output/limitrange06_99" + appender + std::to_string(range) + ".flo";

		stereotgv->initialize(stereoWidth, stereoHeight, beta, gamma, alpha0, alpha1,
			timeStepLambda, lambda, nLevel, fScale, nWarpIters, nSolverIters);
		stereotgv->visualizeResults = true;

		stereotgv->copyMaskToDevice(fisheyeMask);
		stereotgv->loadVectorFields(translationVector, calibrationVector);
		cv::Mat equi1, equi2;
		cv::equalizeHist(im1, equi1);
		cv::equalizeHist(im2, equi2);
		//stereotgv->copyImagesToDevice(im1, im2);
		stereotgv->copyImagesToDevice(equi1, equi2);
		clock_t start = clock();
		stereotgv->solveStereoForwardMasked();
		cv::Mat disparity = cv::Mat(stereoHeight, stereoWidth, CV_32FC2);
		stereotgv->copyDisparityToHost(disparity);
		clock_t timeElapsed = (clock() - start);
		std::cout << "time: " << timeElapsed << " ms" << std::endl;
		cv::writeOpticalFlow(outputFilename, disparity);
		delete stereotgv;
	}
}

int test_LimitingRange() {
	//std::string folder = "C:/Users/cvl-menandro/Downloads/rpg_urban_blender.tar/rpg_urban_blender";
	std::string folder = "D:/dev/blender/icra2020";
	std::string outputFilename = folder + "/output/timing06.flo";
	cv::Mat im1 = cv::imread(folder + "/image/left06.png");
	cv::Mat im2 = cv::imread(folder + "/image/right06.png");
	int stereoWidth = im1.cols;
	int stereoHeight = im1.rows;
	cv::Mat translationVector = cv::readOpticalFlow("D:/dev/matlab_house/translationVectorBlender.flo");
	cv::Mat calibrationVector = cv::Mat::zeros(cv::Size(stereoWidth, stereoHeight), CV_32FC2);
	cv::Mat mask = cv::Mat::zeros(cv::Size(stereoWidth, stereoHeight), CV_8UC1);
	circle(mask, cv::Point(stereoWidth / 2, stereoHeight / 2), stereoWidth / 2 - 50, cv::Scalar(256.0f), -1);
	//cv::imwrite("maskBlender.png", mask);
	//return 0;
	/*cv::imshow("mask", mask);
	std::cout << (int)mask.at<unsigned char>(400, 400) << std::endl;
	cv::waitKey();*/
	cv::Mat fisheyeMask;
	mask.convertTo(fisheyeMask, CV_32F, 1.0 / 255.0);

	StereoTgv * stereotgv = new StereoTgv();
	int width = 800;
	int height = 800;
	float stereoScaling = 1.0f;
	int nLevel = 5;
	float fScale = 2.0;
	int nWarpIters = 1;
	int nSolverIters = 10;
	float lambda = 5.0;
	stereotgv->limitRange = 0.2f;


	float beta = 9.0f;
	float gamma = 0.85f;
	float alpha0 = 17.0f;
	float alpha1 = 1.2f;
	float timeStepLambda = 1.0f;

	stereotgv->initialize(stereoWidth, stereoHeight, beta, gamma, alpha0, alpha1,
		timeStepLambda, lambda, nLevel, fScale, nWarpIters, nSolverIters);
	stereotgv->visualizeResults = true;

	stereotgv->copyMaskToDevice(fisheyeMask);
	stereotgv->loadVectorFields(translationVector, calibrationVector);
	stereotgv->copyImagesToDevice(im1, im2);
	clock_t start = clock();
	stereotgv->solveStereoForwardMasked();
	cv::Mat disparityVis = cv::Mat(stereoHeight, stereoWidth, CV_32FC3);
	//stereotgv->copyDisparityVisToHost(disparityVis, 50.0f);
	//cv::imshow("flow", disparityVis);
	cv::Mat disparity = cv::Mat(stereoHeight, stereoWidth, CV_32FC2);
	stereotgv->copyDisparityToHost(disparity);
	clock_t timeElapsed = (clock() - start);
	std::cout << "time: " << timeElapsed << " ms" << std::endl;
	cv::writeOpticalFlow(outputFilename, disparity);
	// convert disparity to 3D (depends on the model)

	cv::Mat depthVis;
	cv::Mat depth = cv::Mat(stereoHeight, stereoWidth, CV_32F);
	stereotgv->copyStereoToHost(depth);
	depth.copyTo(depthVis, mask);
	showDepthJet("color", depthVis, 5.0f, false);

	cv::Mat warped;
	stereotgv->copyWarpedImageToHost(warped);
	cv::imshow("left", im1);
	cv::imshow("right", im2);
	cv::imshow("warped", warped);
	cv::waitKey();
}

int test_FaroData() {
	//std::string folder = "C:/Users/cvl-menandro/Downloads/rpg_urban_blender.tar/rpg_urban_blender";
	std::string folder = "h:/data_icra/";
	std::string filename = "im146";
	std::string outputFilename = folder + "output/" + filename + ".flo";
	cv::Mat im1 = cv::imread(folder + "image_02/data/" + filename + ".png");
	cv::Mat im2 = cv::imread(folder + "image_03/data/" + filename + ".png");
	cv::Mat equi1, equi2;
	cv::cvtColor(im1, im1, cv::COLOR_BGR2GRAY);
	cv::cvtColor(im2, im2, cv::COLOR_BGR2GRAY);
	cv::equalizeHist(im1, equi1);
	cv::equalizeHist(im2, equi2);
	int stereoWidth = im1.cols;
	int stereoHeight = im1.rows;
	cv::Mat translationVector = cv::readOpticalFlow(folder + "translationVector/" + filename + ".flo");
	cv::Mat calibrationVector = cv::readOpticalFlow(folder + "calibrationVector/" + filename + ".flo");
	cv::Mat mask = cv::Mat::zeros(cv::Size(stereoWidth, stereoHeight), CV_8UC1);
	circle(mask, cv::Point(stereoWidth / 2, stereoHeight / 2), stereoWidth / 2 - 10, cv::Scalar(256.0f), -1);
	//cv::imwrite("maskBlender.png", mask);
	//return 0;
	/*cv::imshow("mask", mask);
	std::cout << (int)mask.at<unsigned char>(400, 400) << std::endl;
	cv::waitKey();*/
	cv::Mat fisheyeMask;
	mask.convertTo(fisheyeMask, CV_32F, 1.0 / 255.0);

	StereoTgv * stereotgv = new StereoTgv();
	int width = 800;
	int height = 800;
	float stereoScaling = 1.0f;
	int nLevel = 11;
	float fScale = 1.2f;
	int nWarpIters = 50;
	int nSolverIters = 50;
	float lambda = 5.0f;
	stereotgv->limitRange = 0.1f;


	float beta = 9.0f;//4.0f;
	float gamma = 0.85f;// 0.2f;
	float alpha0 = 17.0f;// 5.0f;
	float alpha1 = 1.2f;// 1.0f;
	float timeStepLambda = 1.0f;

	stereotgv->initialize(stereoWidth, stereoHeight, beta, gamma, alpha0, alpha1,
		timeStepLambda, lambda, nLevel, fScale, nWarpIters, nSolverIters);
	stereotgv->visualizeResults = true;

	stereotgv->copyMaskToDevice(fisheyeMask);
	stereotgv->loadVectorFields(translationVector, calibrationVector);

	stereotgv->copyImagesToDevice(equi1, equi2);
	stereotgv->solveStereoForwardMasked();


	cv::Mat disparityVis = cv::Mat(stereoHeight, stereoWidth, CV_32FC3);
	stereotgv->copyDisparityVisToHost(disparityVis, 50.0f);
	cv::imshow("flow", disparityVis);

	cv::Mat disparity = cv::Mat(stereoHeight, stereoWidth, CV_32FC2);
	stereotgv->copyDisparityToHost(disparity);
	cv::writeOpticalFlow(outputFilename, disparity);
	// convert disparity to 3D (depends on the model)

	cv::Mat depthVis;
	cv::Mat depth = cv::Mat(stereoHeight, stereoWidth, CV_32F);
	stereotgv->copyStereoToHost(depth);
	depth.copyTo(depthVis, mask);
	//showDepthJet("color", depthVis, 5.0f, false);

	cv::Mat warped;
	stereotgv->copyWarpedImageToHost(warped);
	cv::imshow("right", equi2);
	cv::imshow("left", equi1);
	cv::imshow("warped", warped);
	cv::waitKey();
}

int test_BlenderData() {
	//std::string folder = "C:/Users/cvl-menandro/Downloads/rpg_urban_blender.tar/rpg_urban_blender";
	std::string folder = "h:/blender/icra2020";
	std::string outputFilename = folder + "/output/output04.flo";
	cv::Mat im1 = cv::imread(folder + "/image/left04.png", cv::IMREAD_GRAYSCALE);
	cv::Mat im2 = cv::imread(folder + "/image/right04.png", cv::IMREAD_GRAYSCALE);
	cv::Mat translationVector = cv::readOpticalFlow(folder + "/translationVectorBlender.flo");
	int stereoWidth = im1.cols;
	int stereoHeight = im1.rows;
	
	cv::Mat calibrationVector = cv::Mat::zeros(cv::Size(stereoWidth, stereoHeight), CV_32FC2);
	cv::Mat mask = cv::Mat::zeros(cv::Size(stereoWidth, stereoHeight), CV_8UC1);
	circle(mask, cv::Point(stereoWidth / 2, stereoHeight / 2), stereoWidth/2-50, cv::Scalar(256.0f), -1);
	//cv::imwrite("maskBlender.png", mask);
	//return 0;
	/*cv::imshow("mask", mask);
	std::cout << (int)mask.at<unsigned char>(400, 400) << std::endl;
	cv::waitKey();*/
	cv::Mat fisheyeMask;
	mask.convertTo(fisheyeMask, CV_32F, 1.0 / 255.0);

	StereoTgv * stereotgv = new StereoTgv();
	int width = 800;
	int height = 800;
	float stereoScaling = 1.0f;
	int nLevel = 11; // 11 in paper
	float fScale = 1.2f;
	int nWarpIters = 50;
	int nSolverIters = 50;
	float lambda = 5.0f;
	stereotgv->limitRange = 0.1f;


	float beta = 9.0f;
	float gamma = 0.85f;
	float alpha0 = 17.0f;
	float alpha1 = 1.2f;
	float timeStepLambda = 1.0f;
	
	stereotgv->initialize(stereoWidth, stereoHeight, beta, gamma, alpha0, alpha1,
		timeStepLambda, lambda, nLevel, fScale, nWarpIters, nSolverIters);
	stereotgv->visualizeResults = true;

	stereotgv->copyMaskToDevice(fisheyeMask);
	stereotgv->loadVectorFields(translationVector, calibrationVector);
	cv::Mat equi1, equi2;
	cv::equalizeHist(im1, equi1);
	cv::equalizeHist(im2, equi2);
	//stereotgv->copyImagesToDevice(im1, im2);
	stereotgv->copyImagesToDevice(equi1, equi2);
	stereotgv->solveStereoForwardMasked();

	cv::Mat disparityVis = cv::Mat(stereoHeight, stereoWidth, CV_32FC3);
	stereotgv->copyDisparityVisToHost(disparityVis, 50.0f);
	cv::imshow("flow", disparityVis);

	cv::Mat disparity = cv::Mat(stereoHeight, stereoWidth, CV_32FC2);
	stereotgv->copyDisparityToHost(disparity);
	cv::writeOpticalFlow(outputFilename, disparity);
	// convert disparity to 3D (depends on the model)

	cv::Mat depthVis;
	cv::Mat depth = cv::Mat(stereoHeight, stereoWidth, CV_32F);
	stereotgv->copyStereoToHost(depth);
	depth.copyTo(depthVis, mask);
	showDepthJet("color", depthVis, 5.0f, false);

	cv::Mat warped;
	stereotgv->copyWarpedImageToHost(warped);
	cv::imshow("left", im1);
	cv::imshow("right", im2);
	cv::imshow("warped", warped);
	cv::waitKey();
}

int test_IcraAddedAccuratePixels() {
	std::cout << CV_VER_NUM << std::endl;
	cv::Mat im1 = cv::imread("fs1.png", cv::IMREAD_GRAYSCALE);
	cv::Mat im2 = cv::imread("fs2.png", cv::IMREAD_GRAYSCALE);

	StereoTgv * stereotgv = new StereoTgv();
	int width = 848;
	int height = 800;
	float stereoScaling = 1.0f;
	int nLevel = 7;
	float fScale = 1.5f;
	int nWarpIters = 50;
	int nSolverIters = 50;
	float lambda = 3.0f;
	stereotgv->limitRange = 0.2f;

	int stereoWidth = (int)((float)width / stereoScaling);
	int stereoHeight = (int)((float)height / stereoScaling);
	stereotgv->baseline = 0.0642f;
	stereotgv->focal = 285.8557f / stereoScaling;
	cv::Mat translationVector, calibrationVector;
	if (stereoScaling == 2.0f) {
		translationVector = cv::readOpticalFlow("../test_rstracking/translationVectorHalf.flo");
		calibrationVector = cv::readOpticalFlow("../test_rstracking/calibrationVectorHalf.flo");
	}
	else {
		translationVector = cv::readOpticalFlow("../test_rstracking/translationVector.flo");
		calibrationVector = cv::readOpticalFlow("../test_rstracking/calibrationVector.flo");
	}

	float beta = 4.0f;
	float gamma = 0.2f;
	float alpha0 = 5.0f;
	float alpha1 = 1.0f;
	float timeStepLambda = 1.0f;

	stereotgv->initialize(stereoWidth, stereoHeight, beta, gamma, alpha0, alpha1,
		timeStepLambda, lambda, nLevel, fScale, nWarpIters, nSolverIters);
	stereotgv->visualizeResults = true;

	// Load fisheye Mask
	cv::Mat fisheyeMask8;
	if (stereoScaling == 2.0f) {
		fisheyeMask8 = cv::imread("maskHalf.png", cv::IMREAD_GRAYSCALE);
	}
	else {
		fisheyeMask8 = cv::imread("mask.png", cv::IMREAD_GRAYSCALE);
	}
	cv::Mat fisheyeMask;
	fisheyeMask8.convertTo(fisheyeMask, CV_32F, 1.0 / 255.0);
	//cv::imshow("fm", fisheyeMask);
	stereotgv->copyMaskToDevice(fisheyeMask);

	// Load vector fields
	stereotgv->loadVectorFields(translationVector, calibrationVector);

	//cv::Mat depthVisMask = cv::Mat::zeros(cv::Size(stereoWidth, stereoHeight), CV_8UC1);
	//circle(depthVisMask, cv::Point(stereoWidth / 2, stereoHeight / 2), (int)((float)stereoWidth / 2.2f), cv::Scalar(256.0f), -1);

	// Solve stereo depth
	cv::Mat halfFisheye1, halfFisheye2;
	cv::Mat equi1, equi2;
	cv::equalizeHist(im1, equi1);
	cv::equalizeHist(im2, equi2);

	/*cv::Mat i1fixed = cv::imread("fs1.png", cv::IMREAD_GRAYSCALE);
	cv::Mat i2fixed = cv::imread("fs2.png", cv::IMREAD_GRAYSCALE);
	cv::equalizeHist(i1fixed, equi1);
	cv::equalizeHist(i2fixed, equi2);*/
	cv::imshow("equi", equi1);
	cv::resize(equi1, halfFisheye1, cv::Size(stereoWidth, stereoHeight));
	cv::resize(equi2, halfFisheye2, cv::Size(stereoWidth, stereoHeight));
	//cv::resize(t265.fisheye1, halfFisheye1, cv::Size(stereoWidth, stereoHeight));
	//cv::resize(t265.fisheye2, halfFisheye2, cv::Size(stereoWidth, stereoHeight));

	//std::cout << (int)halfFisheye1.at<uchar>(200, 200) << std::endl;
	stereotgv->copyImagesToDevice(halfFisheye1, halfFisheye2);
	//stereotgv->solveStereoForward();
	stereotgv->solveStereoForwardMasked();

	cv::Mat depth = cv::Mat(stereoHeight, stereoWidth, CV_32F);
	cv::Mat depthVis;
	stereotgv->copyStereoToHost(depth);
	depth.copyTo(depthVis, fisheyeMask8);
	showDepthJet("color", depthVis, 5.0f, false);

	cv::Mat warped;
	stereotgv->copyWarpedImageToHost(warped);
	cv::imshow("warped", warped);

	cv::Mat warped8;
	warped.convertTo(warped8, CV_8U, 256.0);
	cv::imwrite("resim.png", equi1);
	cv::imwrite("resiw.png", warped8);
	saveDepthJet("resdepth.png", depthVis, 5.0f);
	cv::waitKey();
	return 0;
}

void showDepthJet(std::string windowName, cv::Mat image, float maxDepth, bool shouldWait = true) {
	cv::Mat u_norm, u_gray, u_color;
	u_norm = image * 255.0f / maxDepth;
	u_norm.convertTo(u_gray, CV_8UC1);
	cv::applyColorMap(u_gray, u_color, cv::COLORMAP_JET);

	cv::imshow(windowName, u_color);
	if (shouldWait) cv::waitKey();
}

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

void saveDepthJetExponential(std::string fileName, cv::Mat image, float maxDepth, float curve) {
	cv::Mat u_norm, u_exp, u_gray, u_color;
	u_norm = image / maxDepth;
	//std::cout << u_norm.at<float>(400, 400) << std::endl;
	//float curver = 0.1f;
	u_exp = 1.0f - curve / (curve + u_norm);
	u_exp = u_exp * 255.0f;
	u_exp.convertTo(u_gray, CV_8UC1);
	cv::applyColorMap(u_gray, u_color, cv::COLORMAP_JET);

	cv::imwrite(fileName, u_color);
}

void saveDepthJet(std::string fileName, cv::Mat image, float maxDepth) {
	cv::Mat u_norm, u_gray, u_color;
	u_norm = image * 255.0f / maxDepth;
	u_norm.convertTo(u_gray, CV_8UC1);
	cv::applyColorMap(u_gray, u_color, cv::COLORMAP_JET);

	cv::imwrite(fileName, u_color);
}

int test_TwoImagesRealsense() {
	cv::Mat im1 = cv::imread("h:/data_rs_iis/20190913_1/colored_0/data/im174.png", cv::IMREAD_GRAYSCALE);
	cv::Mat im2 = cv::imread("h:/data_rs_iis/20190913_1/colored_1/data/im174.png", cv::IMREAD_GRAYSCALE);

	StereoTgv * stereotgv = new StereoTgv();
	int width = 848;
	int height = 800;
	float stereoScaling = 1.0f;
	int nLevel = 11;
	float fScale = 1.2f;
	int nWarpIters = 50;
	int nSolverIters = 50;
	float lambda = 5.0f;

	int stereoWidth = (int)((float)width / stereoScaling);
	int stereoHeight = (int)((float)height / stereoScaling);
	stereotgv->baseline = 0.0642f;
	stereotgv->focal = 285.8557f / stereoScaling;
	cv::Mat translationVector, calibrationVector;
	if (stereoScaling == 2.0f) {
		translationVector = cv::readOpticalFlow("../test_rstracking/translationVectorHalf.flo");
		calibrationVector = cv::readOpticalFlow("../test_rstracking/calibrationVectorHalf.flo");
	}
	else {
		translationVector = cv::readOpticalFlow("../test_rstracking/translationVector.flo");
		calibrationVector = cv::readOpticalFlow("../test_rstracking/calibrationVector.flo");
	}

	float beta = 4.0f;
	float gamma = 0.2f;
	float alpha0 = 17.0f;
	float alpha1 = 1.2f;
	float timeStepLambda = 1.0f;

	stereotgv->initialize(stereoWidth, stereoHeight, beta, gamma, alpha0, alpha1,
		timeStepLambda, lambda, nLevel, fScale, nWarpIters, nSolverIters);
	stereotgv->visualizeResults = true;

	// Load fisheye Mask
	cv::Mat fisheyeMask8;
	if (stereoScaling == 2.0f) {
		fisheyeMask8 = cv::imread("maskHalf.png", cv::IMREAD_GRAYSCALE);
	}
	else {
		fisheyeMask8 = cv::imread("mask.png", cv::IMREAD_GRAYSCALE);
	}
	cv::Mat fisheyeMask;
	fisheyeMask8.convertTo(fisheyeMask, CV_32F, 1.0 / 255.0);
	//cv::imshow("fm", fisheyeMask);
	stereotgv->copyMaskToDevice(fisheyeMask);

	// Load vector fields
	stereotgv->loadVectorFields(translationVector, calibrationVector);

	//cv::Mat depthVisMask = cv::Mat::zeros(cv::Size(stereoWidth, stereoHeight), CV_8UC1);
	//circle(depthVisMask, cv::Point(stereoWidth / 2, stereoHeight / 2), (int)((float)stereoWidth / 2.2f), cv::Scalar(256.0f), -1);

	// Solve stereo depth
	cv::Mat halfFisheye1, halfFisheye2;
	cv::Mat equi1, equi2;
	cv::equalizeHist(im1, equi1);
	cv::equalizeHist(im2, equi2);

	/*cv::Mat i1fixed = cv::imread("fs1.png", cv::IMREAD_GRAYSCALE);
	cv::Mat i2fixed = cv::imread("fs2.png", cv::IMREAD_GRAYSCALE);
	cv::equalizeHist(i1fixed, equi1);
	cv::equalizeHist(i2fixed, equi2);*/
	//cv::imshow("equi", equi1);
	cv::resize(equi1, halfFisheye1, cv::Size(stereoWidth, stereoHeight));
	cv::resize(equi2, halfFisheye2, cv::Size(stereoWidth, stereoHeight));
	//cv::resize(t265.fisheye1, halfFisheye1, cv::Size(stereoWidth, stereoHeight));
	//cv::resize(t265.fisheye2, halfFisheye2, cv::Size(stereoWidth, stereoHeight));

	cv::Mat depth = cv::Mat(stereoHeight, stereoWidth, CV_32F);
	clock_t start = clock();
	stereotgv->copyImagesToDevice(halfFisheye1, halfFisheye2);
	stereotgv->solveStereoForwardMasked();
	stereotgv->copyStereoToHost(depth);
	clock_t timeElapsed = (clock() - start);

	cv::Mat depthVis;
	std::cout << "time: " << timeElapsed << " ms" << std::endl;
	depth.copyTo(depthVis, fisheyeMask8);
	showDepthJet("color2", depthVis, 30.0f, false);
	cv::waitKey();
	return 0;
}