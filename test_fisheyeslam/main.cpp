#include <slam/fisheyeslam.h>
#include <thread>

int runT265(T265* t265) {
	t265->run();
	return 0;
}

int runSlam(FisheyeSlam* slam) {
	slam->run();
	return 0;
}

int main() {
	T265* t265;
	FisheyeSlam* slam = new FisheyeSlam();

	// Source Device
	t265 = new T265();
	t265->initialize("852212110449");
	std::thread t265Thread(&runT265, t265);

	// SLAM
	slam->initialize(848, 800);
	std::thread slamThread(&runSlam, slam);
	
	cv::Mat im = cv::Mat::zeros(100, 400, CV_8UC3);
	cv::putText(im, "Main fetch thread.", cv::Point(0, 30), cv::FONT_HERSHEY_PLAIN, 2, CV_RGB(255, 255, 255));
	cv::imshow("Main SLAM Thread", im);
	while (1) {
		char pressed = cv::waitKey(10);
		if (pressed == 27) break;

		slam->updateImageSlam(t265->fisheye1, t265->pcXMasked);
		slam->updateImu(t265->ImuRotation);
	}

	t265Thread.join();
	slamThread.join();
	return 0;
}

