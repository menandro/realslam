#include <rslam\rslam.h>

int main() {
	// Tracking using T265
	//Rstracking * rstracking = new Rstracking();
	//rstracking->initialize();
	//rstracking->testFisheye();
	//rstracking->testFeatureDetect();

	// SLAM USING D435i
	Rslam* rslam = new Rslam();
	rslam->initialize(Rslam::D435I_848_480_60);
	rslam->run();
	//rslam->poseSolver();

	/*float x;
	float y;
	float z;
	for (int k = 1; k < 100; k++) {
		rslam->getGyro(&x, &y, &z);
		std::cout << x << " " << y << " " << z << std::endl;
	}*/

	//rslam->testT265();
	//rslam->testStream();
	//rslam->testImu();
	//rslam->showAlignedDepth();
	//rslam->solveRelativePose();
	return 0;
}