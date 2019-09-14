#include "main.h"

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

int test_ImageSequence() {
	std::string mainfolder = "h:/data_rs_iis/20190913_1";
	StereoTgv * stereotgv = new StereoTgv();
	int width = 848;
	int height = 800;
	float stereoScaling = 1.0f;
	int nLevel = 5;
	float fScale = 2.0f;
	int nWarpIters = 4;
	int nSolverIters = 5;
	float lambda = 5.0f;
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
	circle(mask, cv::Point(stereoWidth / 2, stereoHeight / 2), stereoWidth / 2 - 40, cv::Scalar(256.0f), -1);
	cv::Mat fisheyeMask;
	mask.convertTo(fisheyeMask, CV_32F, 1.0 / 255.0);

	stereotgv->copyMaskToDevice(fisheyeMask);

	// Load vector fields
	stereotgv->loadVectorFields(translationVector, calibrationVector);

	cv::Mat halfFisheye1, halfFisheye2;
	cv::Mat equi1, equi2;
	
	for (int k = 0; k <= 1274; k++) {
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
		cv::Mat half1, half2;
		cv::resize(im1, half1, cv::Size(stereoWidth, stereoHeight));
		cv::resize(im2, half2, cv::Size(stereoWidth, stereoHeight));
		//cv::resize(t265.fisheye1, halfFisheye1, cv::Size(stereoWidth, stereoHeight));
		//cv::resize(t265.fisheye2, halfFisheye2, cv::Size(stereoWidth, stereoHeight));
		clock_t start = clock();
		//std::cout << (int)halfFisheye1.at<uchar>(200, 200) << std::endl;
		stereotgv->copyImagesToDevice(halfFisheye1, halfFisheye2);
		//stereotgv->copyImagesToDevice(half1, half2);

		//stereotgv->solveStereoForward();
		stereotgv->solveStereoForwardMasked();

		cv::Mat depth = cv::Mat(stereoHeight, stereoWidth, CV_32F);
		cv::Mat depthVis;
		stereotgv->copyStereoToHost(depth);
		clock_t timeElapsed = (clock() - start);
		std::cout << "time: " << timeElapsed << " ms " ;
		//stereotgv->copy1DDisparityToHost(depth);
		depth.copyTo(depthVis, mask);
		showDepthJet("color", depthVis, 3.0f, false);
		
		std::string appender;
		if (k < 10) appender = "000";
		else if ((k >= 10) && (k < 100)) appender = "00";
		else if ((k >= 100) && (k < 1000)) appender = "0";
		else appender = "";
		saveDepthJet(mainfolder + "/fps1/im" + appender + std::to_string(k) + ".png", depthVis, 5.0f);
		//saveDepthJet("h:/data_rs_iis/20190909/output1ddisparity/im" + std::to_string(k) + ".png", depthVis, 30.0f);
		/*cv::Mat imout;
		im1.copyTo(imout, fisheyeMask8);
		cv::imwrite("d:/data/20190909/smalloutputimage/im" + appender + std::to_string(k) + ".png", imout);*/
		std::cout << k << std::endl;
		cv::imshow("test", halfFisheye1);
		cv::waitKey(1);
	}
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
	std::string folder = "d:/data/data_icra/";
	std::string filename = "im92";
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
	std::string folder = "D:/dev/blender/icra2020";
	std::string outputFilename = folder + "/output/output10.flo";
	cv::Mat im1 = cv::imread(folder + "/image/left10.png", cv::IMREAD_GRAYSCALE);
	cv::Mat im2 = cv::imread(folder + "/image/right10.png", cv::IMREAD_GRAYSCALE);
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
	u_norm = image * 256.0f / maxDepth;
	u_norm.convertTo(u_gray, CV_8UC1);
	cv::applyColorMap(u_gray, u_color, cv::COLORMAP_JET);

	cv::imshow(windowName, u_color);
	if (shouldWait) cv::waitKey();
}

void saveDepthJet(std::string fileName, cv::Mat image, float maxDepth) {
	cv::Mat u_norm, u_gray, u_color;
	u_norm = image * 256.0f / maxDepth;
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

	//std::cout << (int)halfFisheye1.at<uchar>(200, 200) << std::endl;
	stereotgv->copyImagesToDevice(halfFisheye1, halfFisheye2);
	//stereotgv->solveStereoForward();
	stereotgv->solveStereoForwardMasked();

	cv::Mat depth = cv::Mat(stereoHeight, stereoWidth, CV_32F);
	cv::Mat depthVis;
	stereotgv->copyStereoToHost(depth);
	depth.copyTo(depthVis, fisheyeMask8);
	showDepthJet("color", depthVis, 30.0f, false);
	cv::waitKey();
	return 0;
}