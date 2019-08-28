#include "main.h"

int test_FaroData() {
	//std::string folder = "C:/Users/cvl-menandro/Downloads/rpg_urban_blender.tar/rpg_urban_blender";
	std::string folder = "H:/data_icra/";
	std::string outputFilename = "outputfaro1.flo";
	cv::Mat im1 = cv::imread(folder + "image_02/data/im1.png");
	cv::Mat im2 = cv::imread(folder + "image_03/data/im1.png");
	int stereoWidth = im1.cols;
	int stereoHeight = im1.rows;
	cv::Mat translationVector = cv::readOpticalFlow("D:/dev/matlab_house/translationVectorFaro.flo");
	cv::Mat calibrationVector = cv::readOpticalFlow("D:/dev/matlab_house/calibrationVectorFaro.flo");
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
	int nLevel = 14;
	float fScale = 1.2f;
	int nWarpIters = 50;
	int nSolverIters = 50;
	float lambda = 3.0f;
	stereotgv->limitRange = 0.1f;


	float beta = 4.0f;
	float gamma = 0.2f;
	float alpha0 = 5.0f;
	float alpha1 = 1.0f;
	float timeStepLambda = 1.0f;

	stereotgv->initialize(stereoWidth, stereoHeight, beta, gamma, alpha0, alpha1,
		timeStepLambda, lambda, nLevel, fScale, nWarpIters, nSolverIters);
	stereotgv->visualizeResults = true;

	stereotgv->copyMaskToDevice(fisheyeMask);
	stereotgv->loadVectorFields(translationVector, calibrationVector);
	stereotgv->copyImagesToDevice(im1, im2);
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

int test_BlenderData() {
	//std::string folder = "C:/Users/cvl-menandro/Downloads/rpg_urban_blender.tar/rpg_urban_blender";
	std::string folder = "D:/dev/blender/plane";
	std::string outputFilename = "output03.flo";
	cv::Mat im1 = cv::imread(folder + "/image/left03.png");
	cv::Mat im2 = cv::imread(folder + "/image/right03.png");
	int stereoWidth = im1.cols;
	int stereoHeight = im1.rows;
	cv::Mat translationVector = cv::readOpticalFlow("D:/dev/matlab_house/translationVectorBlender.flo");
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
	int nLevel = 14;
	float fScale = 1.2f;
	int nWarpIters = 50;
	int nSolverIters = 50;
	float lambda = 3.0f;
	stereotgv->limitRange = 0.1f;


	float beta = 4.0f;
	float gamma = 0.2f;
	float alpha0 = 5.0f;
	float alpha1 = 1.0f;
	float timeStepLambda = 1.0f;
	
	stereotgv->initialize(stereoWidth, stereoHeight, beta, gamma, alpha0, alpha1,
		timeStepLambda, lambda, nLevel, fScale, nWarpIters, nSolverIters);
	stereotgv->visualizeResults = true;

	stereotgv->copyMaskToDevice(fisheyeMask);
	stereotgv->loadVectorFields(translationVector, calibrationVector);
	stereotgv->copyImagesToDevice(im1, im2);
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
	cv::Mat im1 = cv::imread("fs1.png", cv::IMREAD_GRAYSCALE);
	cv::Mat im2 = cv::imread("fs2.png", cv::IMREAD_GRAYSCALE);

	StereoTgv * stereotgv = new StereoTgv();
	int width = 848;
	int height = 800;
	float stereoScaling = 2.0f;
	int nLevel = 4;
	float fScale = 2.0f;
	int nWarpIters = 20;
	int nSolverIters = 20;
	float lambda = 3.0f;

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
	cv::imshow("fm", fisheyeMask);
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
	cv::waitKey();
	return 0;
}