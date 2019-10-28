#include <rslam\rslam.h>
#include <tslam/Tslam.h>

int main() {
	// ICRA Paper
	Tslam *tslam = new Tslam();
	tslam->initialize("852212110449");
	tslam->run();

	// Save images from recorded t265
	//Tslam *tslam = new Tslam();
	//tslam->saveT265Images("h:/data_rs_iis/20190913_11/bag/20190913_103425.bag", "h:/data_rs_iis/20190913_11/");

	/*double scale = 2.0;
	cv::Mat mask = cv::Mat::zeros(cv::Size(848 / scale, 800/scale), CV_8UC1);
	cv::Mat output;
	cv::circle(mask, cv::Point(421.500/ scale, 402.500 / scale), 382.5552 / scale, cv::Scalar(255, 255, 255), -1);
	cv::imshow("mask", mask);
	cv::imwrite("maskHalf.png", mask);
	cv::waitKey();*/

	//tslam->testStereo("fs1.png", "fs2.png");

	/*Rslam* rslam = new Rslam();
	rslam->testT265();*/

	// SLAM USING D435i-Live
	/*Rslam* rslam = new Rslam();
	rslam->initialize(Rslam::D435I_IR_640_360_90, Rslam::ORB, "843112071357", "841612070674");
	rslam->run();*/

	// SLAM USING D435i-Recorded
	/*Rslam* rslam = new Rslam();
	rslam->initializeFromFile("H:/data_rs_iis/20190710/bag/84311207135711.bag", "h:/data_rs_iis/20190710/bag/85221211044911.bag");
	rslam->runFromRecording();*/

	// SAVE all files
	/*Rslam* rslam = new Rslam();
	std::string fileset = "12";
	rslam->saveAllDepthAndInfrared(std::string("H:/data_rs_iis/20190710/bag/843112071357" + fileset + ".bag").c_str(),
		std::string("h:/data_rs_iis/20190710/bag/852212110449" + fileset + ".bag").c_str(),
		std::string("H:/data_rs_iis/20190710/" + fileset + "/").c_str());*/
	/*rslam->saveImu(std::string("H:/data_rs_iis/20190710/bag/843112071357" + fileset + ".bag").c_str(),
		std::string("h:/data_rs_iis/20190710/bag/852212110449" + fileset + ".bag").c_str(),
		std::string("H:/data_rs_iis/20190710/" + fileset + "/").c_str());*/
	/*rslam->saveExternalImu(std::string("H:/data_rs_iis/20190710/bag/843112071357" + fileset + ".bag").c_str(),
		std::string("h:/data_rs_iis/20190710/bag/852212110449" + fileset + ".bag").c_str(),
		std::string("H:/data_rs_iis/20190710/" + fileset + "/").c_str());*/
	/*rslam->getSynchronization(std::string("H:/data_rs_iis/20190710/bag/843112071357" + fileset + ".bag").c_str(),
		std::string("h:/data_rs_iis/20190710/bag/852212110449" + fileset + ".bag").c_str(),
		std::string("H:/data_rs_iis/20190710/" + fileset + "/").c_str());*/

	/*rslam->saveAllFramesFinal("H:/data_rs_iis/20190710/bag/84311207135711.bag", 
		"h:/data_rs_iis/20190710/bag/85221211044911.bag", 
		"H:/data_rs_iis/20190710/");*/

	/*rslam->getSynchronization("H:/data_rs_iis/20190710/bag/84311207135711.bag",
		"h:/data_rs_iis/20190710/bag/85221211044911.bag",
		"H:/data_rs_iis/20190710/");*/

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