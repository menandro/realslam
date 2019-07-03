#include <rslam\rslam.h>

int main() {
	// SLAM USING D435i-Live
	/*Rslam* rslam = new Rslam();
	rslam->initialize(Rslam::D435I_IR_640_360_90, Rslam::ORB, "843112071357", "841612070674");
	rslam->run();*/

	// SLAM USING D435i-Recorded
	Rslam* rslam = new Rslam();
	rslam->initializeFromFile("H:/data_rs_iis/8431120713573.bag", "h:/data_rs_iis/8522121104493.bag");
	rslam->runFromRecording();

	// Tracking using T265
	//Rstracking * rstracking = new Rstracking();
	//rstracking->initialize();
	//rstracking->testFisheye();
	//rstracking->testFeatureDetect();


	/*std::thread t1(&Rslam::visualizePose, rslam);
	std::thread t3(&Rslam::poseSolverDefaultStereoMulti, rslam);

	t1.join();
	t3.join();*/

	//rslam->testOrb();
	//rslam->recordAll();
	//rslam->playback("843112071357");
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