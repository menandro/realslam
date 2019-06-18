#include "Rstracking.h"

int Rstracking::initialize() {
	width = 848; //Fixed
	height = 800;
	cameraLeft = cv::Mat(cv::Size(width, height), CV_8UC1);
	cameraRight = cv::Mat(cv::Size(width, height), CV_8UC1);
	double kLeft[9] = { 285.722, 0.0, 420.135, 0.0, 286.759, 403.394, 0.0, 0.0, 1.0 };
	intrinsicLeft = cv::Mat(3, 3, CV_64FC1, kLeft).clone();

	double kRight[9] = { 284.936, 0.0, 428.136, 0.0, 286.006, 398.921, 0.0, 0.0, 1.0 };
	intrinsicRight = cv::Mat(3, 3, CV_64FC1, kRight).clone();

	double dLeft[4] = { -0.00659769, 0.0473251, -0.0458264, 0.00897725 };
	distortionLeft = cv::Mat(1, 4, CV_64FC1, dLeft).clone();

	double dRight[4] = { -0.00492777, 0.0391601, -0.0353147, 0.0051312 };
	distortionRight = cv::Mat(1, 4, CV_64FC1, dRight).clone();

	try {
		ctx = new rs2::context();
		pipe = new rs2::pipeline(*ctx);
		rs2::config cfg;
		auto dev = ctx->query_devices();
		cfg.enable_device("852212110449");
		pipe->start(cfg);
		return EXIT_SUCCESS;
	}
	catch (const rs2::error & e)
	{
		std::cerr << "RealSense error calling " << e.get_failed_function() << "(" << e.get_failed_args() << "):\n    " << e.what() << std::endl;
		return EXIT_FAILURE;
	}
	catch (const std::exception& e)
	{
		std::cerr << e.what() << std::endl;
		return EXIT_FAILURE;
	}
	return 0;
}

int Rstracking::testFeatureDetect() {
	CameraPose * camerapose = new CameraPose();
	camerapose->initialize(285.722, 286.759, cv::Point2d(420.145, 403.394), 2000);
	cv::Mat R, t;
	while (1)
	{
		char pressed = cv::waitKey(10);
		if (pressed == 27) break; //press escape

		rs2::frameset frameset = pipe->wait_for_frames();
		auto fisheye1 = frameset.get_fisheye_frame(1);
		auto fisheye2 = frameset.get_fisheye_frame(2);
		cameraLeft = cv::Mat(cv::Size(width, height), CV_8UC1, (void*)fisheye1.get_data(), cv::Mat::AUTO_STEP);
		cameraRight = cv::Mat(cv::Size(width, height), CV_8UC1, (void*)fisheye2.get_data(), cv::Mat::AUTO_STEP);

		//cv::undistort(cameraLeft, undistortLeft, intrinsicLeft, distortionLeft);
		//cv::fisheye::undistortImage(cameraLeft, undistortLeft, intrinsicLeft, distortionLeft);
		//cv::remap(cameraLeft, undistortLeft, map1, map2, CV_INTER_CUBIC);
		//cv::imshow("right", undistortRight);
		//cv::imshow("left", undistortLeft);
		camerapose->solvePose_8uc1(cameraLeft, cameraRight, R, t);
		camerapose->drawKeypoints();
	}
	pipe->stop();
	return 0;
}

int Rstracking::testFeatureDetectUndistort() {
	CameraPose * camerapose = new CameraPose();
	camerapose->initialize(285.722, 286.759, cv::Point2d(420.145, 403.394), 2000);
	cv::Mat R, t;
	cv::Mat map1, map2;
	cv::Mat undistortLeft, undistortRight;// = cv::Mat::zeros(200, 212, CV_8UC1);
	cv::Mat newIntrinsicLeft = intrinsicLeft.clone();
	cv::Mat newIntrinsicRight = intrinsicRight.clone();

	//cv::fisheye::initUndistortRectifyMap(intrinsicLeft, distortionLeft, cv::Mat::eye(3,3,CV_64F), intrinsicLeft, cv::Size(848,800), CV_8UC1, map1, map2);
	cv::fisheye::estimateNewCameraMatrixForUndistortRectify(intrinsicLeft, distortionLeft, cameraLeft.size(), cv::noArray(), newIntrinsicLeft, 1.0);
	cv::fisheye::estimateNewCameraMatrixForUndistortRectify(intrinsicRight, distortionRight, cameraRight.size(), cv::noArray(), newIntrinsicRight, 1.0);

	int newFocalLength = 300;
	newIntrinsicLeft.at<double>(0, 0) = newFocalLength;
	newIntrinsicLeft.at<double>(1, 1) = newFocalLength;
	newIntrinsicRight.at<double>(0, 0) = newFocalLength;
	newIntrinsicRight.at<double>(1, 1) = newFocalLength;
	/*std::cout << intrinsicLeft << std::endl;
	std::cout << newIntrinsicLeft << std::endl;
	std::cout << distortionLeft << std::endl;*/
	while (1)
	{
		char pressed = cv::waitKey(10);
		if (pressed == 27) break; //press escape

		rs2::frameset frameset = pipe->wait_for_frames();
		auto fisheye1 = frameset.get_fisheye_frame(1);
		auto fisheye2 = frameset.get_fisheye_frame(2);
		cameraLeft = cv::Mat(cv::Size(width, height), CV_8UC1, (void*)fisheye1.get_data(), cv::Mat::AUTO_STEP);
		cameraRight = cv::Mat(cv::Size(width, height), CV_8UC1, (void*)fisheye2.get_data(), cv::Mat::AUTO_STEP);

		cv::fisheye::undistortImage(cameraLeft, undistortLeft, intrinsicLeft, distortionLeft, newIntrinsicLeft);
		cv::fisheye::undistortImage(cameraRight, undistortRight, intrinsicRight, distortionRight, newIntrinsicRight);

		//cv::undistort(cameraLeft, undistortLeft, intrinsicLeft, distortionLeft);
		//cv::fisheye::undistortImage(cameraLeft, undistortLeft, intrinsicLeft, distortionLeft);
		//cv::remap(cameraLeft, undistortLeft, map1, map2, CV_INTER_CUBIC);
		//cv::imshow("right", undistortRight);
		//cv::imshow("left", undistortLeft);
		camerapose->solvePose_8uc1(undistortLeft, undistortRight, R, t);
		camerapose->drawKeypoints();
	}
	pipe->stop();
	return 0;
}

int Rstracking::testFisheye() {
	cv::namedWindow("cameraLeft", cv::WINDOW_AUTOSIZE);
	cv::namedWindow("cameraRight", cv::WINDOW_AUTOSIZE);

	while (cv::waitKey(1) < 0 && cv::getWindowProperty("cameraLeft", cv::WND_PROP_AUTOSIZE) >= 0)
	{
		rs2::frameset frameset = pipe->wait_for_frames();
		auto fisheye1 = frameset.get_fisheye_frame(1);
		auto fisheye2 = frameset.get_fisheye_frame(2);
		cameraLeft = cv::Mat(cv::Size(width, height), CV_8UC1, (void*)fisheye1.get_data(), cv::Mat::AUTO_STEP);
		cameraRight = cv::Mat(cv::Size(width, height), CV_8UC1, (void*)fisheye2.get_data(), cv::Mat::AUTO_STEP);
		cv::imshow("cameraLeft", cameraLeft);
		cv::imshow("cameraRight", cameraRight);
	}
	return 0;
}