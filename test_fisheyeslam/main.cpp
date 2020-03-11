#include <slam/fisheyeslam.h>
#include <thread>

int main() {
	T265* t265;
	FisheyeSlam* slam = new FisheyeSlam();
	int width = 424;
	int height = 400;

	// Source Device
	t265 = new T265();
	float stereoScaling = 2.0f;
	t265->initialize("852212110449", stereoScaling);
	t265->run();
	std::cout << "T265 Running." << std::endl;

	// SLAM
	cv::Mat fisheyeMask = cv::imread("maskHalf.png", cv::IMREAD_GRAYSCALE);
	slam->initialize(width, height, t265->intrinsic, t265->distCoeffs, fisheyeMask);
	slam->run();
	
	cv::Mat im = cv::Mat::zeros(100, 400, CV_8UC3);
	cv::putText(im, "Main fetch thread.", cv::Point(0, 30), cv::FONT_HERSHEY_PLAIN, 2, CV_RGB(255, 255, 255));
	cv::imshow("Main SLAM Thread", im);
	while (1) {
		char pressed = cv::waitKey(10);
		if (pressed == 27) break;

		slam->updateImageSlam(t265->image, t265->X);
		slam->updateImu(t265->ImuRotation);
	}
	return 0;
}

