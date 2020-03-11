#include "fisheyeslam.h"

FisheyeSlam::FisheyeSlam() {

}

int FisheyeSlam::initialize(int width, int height) {
	this->height = height;
	this->width = width;

	initOpticalFlow();

	return 0;
}

int FisheyeSlam::updateImageSlam(cv::Mat im, cv::Mat point3d) {
	currImage = im;
	currPoint3d = point3d;
	isImageUpdated = true;
	return 0;
}

int FisheyeSlam::updateImu(Quaternion imuRotation) {
	this->imuRotation = imuRotation;
	isImuUpdated = true;
	return 0;
}

int FisheyeSlam::run() {
	std::thread t0(&FisheyeSlam::tracking, this);
	t0.join();
	return 0;
}

int FisheyeSlam::tracking() {
	cv::Mat im = cv::Mat::zeros(100, 400, CV_8UC3);
	cv::putText(im, "Tracking Thread", cv::Point(0, 30), cv::FONT_HERSHEY_PLAIN, 2, CV_RGB(255, 255, 255));
	cv::imshow("Tracking Thread", im);
	while (1) {
		char pressed = cv::waitKey(10);
		if (pressed == 27) break;

		if (!isImageUpdated) continue;

		if (!keyframeExist) {
			currKfImage = currImage.clone();
			keyframeExist = true;
		}
		else {
			cv::imshow("im1", currImage);
			cv::imshow("im2", prevImage);
			solveOpticalFlow(prevImage, currImage);
			prevImage = currImage.clone();
		}
		isImageUpdated = false;
	}
	
	return 0;
}

int FisheyeSlam::mapping() {

	return 0;
}