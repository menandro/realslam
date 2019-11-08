#include "Tslam.h"

int Tslam::initialize(const char* serialNumber) {
	viewer = new Viewer();

	// Fisheye Stereo
	//initStereoTVL1();
	initStereoTGVL1();

	// Upsampling
	initDepthUpsampling();

	try {
		ctx = new rs2::context();
		auto dev = ctx->query_devices();

		for (auto&& devfound : dev) {
			const char * serialNo = devfound.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER);
			std::cout << "Found device: " << serialNo << std::endl;
			std::vector<rs2::sensor> sensors = devfound.query_sensors();
			sensors[0].set_option(RS2_OPTION_ENABLE_AUTO_EXPOSURE, 0);
			sensors[0].set_option(RS2_OPTION_EXPOSURE, 16000);
			sensors[0].set_option(RS2_OPTION_GAIN, 1);

			// Create pipeline for device0
			if (isThisDevice(serialNo, serialNumber)) {
				t265.serialNo = serialNumber;
				t265.pipe = new rs2::pipeline(*ctx);

				std::cout << "Configuring " << serialNo << std::endl;
				t265.cfg.enable_stream(RS2_STREAM_FISHEYE, 1, 848, 800, rs2_format::RS2_FORMAT_Y8, 30);
				t265.cfg.enable_stream(RS2_STREAM_FISHEYE, 2, 848, 800, rs2_format::RS2_FORMAT_Y8, 30);
				t265.cfg.enable_stream(RS2_STREAM_ACCEL, RS2_FORMAT_MOTION_XYZ32F, 62);
				t265.cfg.enable_stream(RS2_STREAM_GYRO, RS2_FORMAT_MOTION_XYZ32F, 200);
				
				auto prof = t265.pipe->start(t265.cfg);
				// Disable auto-exposure
				

				t265.isFound = true;
				std::cout << "Pipe created from: " << serialNo << std::endl;
			}
		}
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

	matcher = cv::cuda::DescriptorMatcher::createBFMatcher(cv::NORM_HAMMING); //for orb
	orb = cv::cuda::ORB::create(500, 2.0f, 2, 10, 0, 2, 0, 10);
	orb->setBlurForDescriptor(true);

	t265.fisheye1 = cv::Mat(t265.height, t265.width, CV_8UC1);
	t265.fisheye2 = cv::Mat(t265.height, t265.width, CV_8UC1);
	t265.fisheye132f = cv::Mat(t265.height, t265.width, CV_32F);
	t265.fisheye232f = cv::Mat(t265.height, t265.width, CV_32F);
}

int Tslam::initDepthUpsampling() {
	upsampling = new lup::Upsampling(32, 12, 32);
	int maxIter = 100;
	float beta = 9.0f;
	float gamma = 0.85f;
	float alpha0 = 17.0f;
	float alpha1 = 1.2f;
	float timeStepLambda = 1.0f;
	float lambdaTgvl2 = 0.1f;
	float maxUpsamplingDepth = 5.0f;
	this->maxUpsamplingDepth = maxUpsamplingDepth;
	upsampling->initialize(stereoWidth, stereoHeight, maxIter, beta, gamma,
		alpha0, alpha1, timeStepLambda, lambdaTgvl2, maxUpsamplingDepth);
	return 0;
}

int Tslam::run() {
	//std::thread t1(&Rslam::visualizePose, this);
	//std::thread t2(&Rslam::poseSolver, this);
	//std::thread t2(&Rslam::poseSolverDefaultStereo, this);
	std::thread t0(&Tslam::fetchFrames, this);
	std::thread t1(&Tslam::imuPoseSolver, this);
	//std::thread t2(&Rslam::poseSolverDefaultStereoMulti, this);
	std::thread t2(&Tslam::cameraPoseSolver, this);
	std::thread t3(&Tslam::visualizePose, this);

	t0.join();
	t1.join();
	t2.join();
	t3.join();

	t265.pipe->stop();
	return 0;
}

int Tslam::fetchFrames() {
	cv::Mat im = cv::Mat::zeros(100, 400, CV_8UC3);
	cv::putText(im, "Main fetch thread.", cv::Point(0, 30), cv::FONT_HERSHEY_PLAIN, 2, CV_RGB(255, 255, 255));
	cv::imshow("Main", im);
	while (true) {
		char pressed = cv::waitKey(10);
		if (pressed == 27) break;

		// Poll framesets multi-camera (when any is available)
		//if (!mutex.try_lock()) continue;
		bool pollSuccess = (t265.pipe->poll_for_frames(&t265.frameset));
		//mutex.unlock();
		//device0.frameset = device0.pipe->wait_for_frames();
		if (!pollSuccess) continue;

		auto gyroFrame = t265.frameset.first_or_default(RS2_STREAM_GYRO, RS2_FORMAT_MOTION_XYZ32F).as<rs2::motion_frame>();
		auto accelFrame = t265.frameset.first_or_default(RS2_STREAM_ACCEL, RS2_FORMAT_MOTION_XYZ32F).as<rs2::motion_frame>();
		auto fisheye1Data = t265.frameset.get_fisheye_frame(1);
		auto fisheye2Data = t265.frameset.get_fisheye_frame(2);
		t265.gyroQueue.enqueue(gyroFrame);
		t265.accelQueue.enqueue(accelFrame);

		t265.fisheye1 = cv::Mat(cv::Size(t265.width, t265.height), CV_8UC1, (void*)fisheye1Data.get_data(), cv::Mat::AUTO_STEP);
		t265.fisheye2 = cv::Mat(cv::Size(t265.width, t265.height), CV_8UC1, (void*)fisheye2Data.get_data(), cv::Mat::AUTO_STEP);
		/*cv::imshow("fisheye1", t265.fisheye1);
		cv::imshow("fisheye2", t265.fisheye2);*/

		// test undistortion: CORRECT!!!
		//cv::Mat undistorted;
		//double intrinsic[9] = { 285.722, 0, 420.135, 0, 286.759, 403.394, 0, 0, 1 };
		//double focal = 321.902;// 120;
		////double intrinsicNew[9] = { focal, 0, 320.729, 0, focal,  181.862, 0, 0, 1 };
		//double intrinsicNew[9] = { focal, 0, 320.729, 0, focal,  320.729, 0, 0, 1 };
		////double distortion[4] = { -0.00659769,0.0473251, -0.0458264, 0.00897725};
		//double distortion[5] = { -0.00659769, 0.0473251, -0.0458264, 0.00897725 };
		//cv::Mat intMat = cv::Mat(3, 3, CV_64F, intrinsic).clone();
		//cv::Mat intNewMat = cv::Mat(3, 3, CV_64F, intrinsicNew).clone();
		//cv::Mat distCoeffs = cv::Mat(1, 4, CV_64F, distortion).clone();
		//cv::fisheye::undistortImage(t265.fisheye1, undistorted, intMat, distCoeffs, intNewMat, cv::Size(640,360));
		////cv::undistort(t265.fisheye1, undistorted, intMat, distCoeffs, intNewMat);
		//cv::imshow("fisheyeundist", undistorted);
		////cv::imwrite("fisheyeundist.png", undistorted);
		////cv::imwrite("fisheye.png", t265.fisheye1);

		// Test Optical Flo
		/*stereo->solveOpticalFlow();
		cv::Mat uvrgb = cv::Mat(t265.height, t265.width, CV_32FC3);
		stereo->copyOpticalFlowVisToHost(uvrgb);
		cv::imshow("disparity", uvrgb);*/
	}
	return 0;
}

int Tslam::cameraPoseSolver() {
	t265.currentKeyframe = new Keyframe();
	t265.keyframeExist = false;
	t265.currentKeyframe->R = cv::Mat::zeros(3, 1, CV_64F);
	t265.currentKeyframe->t = cv::Mat::zeros(3, 1, CV_64F);
	t265.currentKeyframe->currentRelativeR = cv::Mat::zeros(3, 1, CV_64F);
	t265.currentKeyframe->currentRelativeT = cv::Mat::zeros(3, 1, CV_64F);

	// Create fisheye mask
	t265.fisheyeMask = cv::Mat::zeros(cv::Size(848, 800), CV_8UC1);
	circle(t265.fisheyeMask, cv::Point(424, 400), 385, cv::Scalar(256.0f), -1);
	//cv::imshow("fisheye mask", t265.fisheyeMask);
	t265.d_fisheyeMask.upload(t265.fisheyeMask);

	while (true) {
		char pressed = cv::waitKey(10);
		if (pressed == 27) break;
		if (pressed == 'r') {
			t265.keyframeExist = false; // change this to automatic keyframing
		}

		//if (!(processDepth(device0) && processIr(device0))) continue;
		//upsampleDepth(device0);
		//visualizeDepth(device0);
		//createDepthThresholdMask(device0, 2.0f);

		// Detect Feature points
		/*detectAndComputeOrb(t265.fisheye1, t265.d_fe1, t265.keypointsFe1, t265.d_descriptorsFe1);
		detectAndComputeOrb(t265.fisheye2, t265.d_fe2, t265.keypointsFe2, t265.d_descriptorsFe2);*/
		/*cv::Mat equi1, equi2;
		cv::equalizeHist(t265.fisheye1, equi1);
		cv::equalizeHist(t265.fisheye2, equi2);
		cv::imwrite("fs1.png", equi1);
		cv::imwrite("fs2.png", equi2);*/
		//visualizeKeypoints(t265, "kp");

		//solveStereoTVL1();
		solveStereoTGVL1();
		createDepthThresholdMask(t265, 1.0f);

		//cv::Mat depthUpsample = cv::Mat(stereoHeight, upsampling->iAlignUp(stereoWidth), CV_32F);
		//cv::Mat depthPad, imagePad;
		//cv::copyMakeBorder(depthVis, depthPad, 0, 0, 0, upsampling->iAlignUp(stereoWidth) - stereoWidth, cv::BORDER_CONSTANT, 0);
		//cv::copyMakeBorder(halfFisheye1, imagePad, 0, 0, 0, upsampling->iAlignUp(stereoWidth) - stereoWidth, cv::BORDER_CONSTANT, 0);
		////std::cout << depth.size() << " " << depthVis.size() << std::endl;
		//upsampling->copyImagesToDevice(imagePad, depthPad);
		////upsampling->propagateColorOnly(10);
		//upsampling->optimizeOnly();
		////upsampling->solve();
		//upsampling->copyImagesToHost(depthUpsample);
		//depthUpsample = depthUpsample * this->maxUpsamplingDepth;
		////std::cout << equi1.at<float>(200, 200) << std::endl;
		//showDepthJet("upsample", depthUpsample, 5.0f, false);

		//std::cout << depth.at<float>(cv::Point(424, 400)) << std::endl;
		/*cv::Mat uvrgb = cv::Mat(t265.height, t265.width, CV_32FC3);
		stereo->copyOpticalFlowVisToHost(uvrgb);
		cv::imshow("flow", uvrgb);*/
		

		/*stereoMatching(t265);
		visualizeMatchedStereoPoints(t265, "stereo");*/

		// Match with keyframe
		//matchAndPose(t265);
		//matchAndPose(device1);
		//visualizeRelativeKeypoints(device0.currentKeyframe, device0.infrared1, "dev0");
		//visualizeRelativeKeypoints(device1.currentKeyframe, device1.infrared1, "dev1");

		/*Device viewDevice = device0;
		viewDevice.Rvec = viewDevice.currentKeyframe->currentRelativeR;
		viewDevice.t = viewDevice.currentKeyframe->currentRelativeT;
		this->Rvec = viewDevice.Rvec;
		this->t = viewDevice.t;
		cv::Mat im = cv::Mat::zeros(100, 300, CV_8UC3);
		im.setTo(cv::Scalar(50, 50, 50));
		overlayMatrix("pose", im, this->Rvec, this->t);

		updateViewerCameraPose(device0);*/
	}
	return 0;
}

int Tslam::detectAndComputeOrb(cv::Mat im, cv::cuda::GpuMat &d_im,
	std::vector<cv::KeyPoint> &keypoints, cv::cuda::GpuMat &descriptors) {
	d_im.upload(im);
	orb->detectAndCompute(d_im, t265.d_fisheyeMask, keypoints, descriptors);
	return 0;
}


int Tslam::matchAndPose(T265& device) {
	if (!device.keyframeExist) {
		device.currentKeyframe->im = device.fisheye1.clone();
		device.currentKeyframe->d_im.upload(device.currentKeyframe->im);
		device.currentKeyframe->keypoints = device.keypointsFe1;
		device.currentKeyframe->d_descriptors = device.d_descriptorsFe1.clone();
		device.keyframeExist = true;
	}
	else {
		//relativeMatchingDefaultStereo(device, device.currentKeyframe, device.infrared1);
		//solveRelativePose(device, device.currentKeyframe);
	}
	return 0;
}

int Tslam::stereoMatching(T265 &device) {
	if ((device.keypointsFe1.empty() || device.keypointsFe2.empty()) || (device.d_descriptorsFe1.cols <= 1) || (device.d_descriptorsFe2.cols <= 1)) {
		std::cout << "No keypoints found." << std::endl;
	}
	else {
		matcher->knnMatch(device.d_descriptorsFe1, device.d_descriptorsFe2, device.stereoMatches, 2);
		if (!device.stereoMatches.empty()) {
			//std::cout << "Matches: " << matches.size() << std::endl;
			device.stereoKeypoints = std::vector< cv::KeyPoint >();
			device.stereoKeypointsSrc = std::vector< cv::KeyPoint >();
			device.stereoKeypoints.clear();
			device.stereoKeypointsSrc.clear();

			device.stereoPoints = std::vector<cv::Point2f>();
			device.stereoPointsSrc = std::vector<cv::Point2f>();
			device.stereoPoints.clear();
			device.stereoPointsSrc.clear();

			device.stereoDistances = std::vector< float >();
			device.stereoDistances.clear();

			device.stereoObjectPointsSrc = std::vector<cv::Point3f>();
			device.stereoObjectPointsSrc.clear();

			for (int k = 0; k < (int)device.stereoMatches.size(); k++)
			{
				if ((device.stereoMatches[k][0].distance < 0.6*(device.stereoMatches[k][1].distance)) &&
					((int)device.stereoMatches[k].size() <= 2 && (int)device.stereoMatches[k].size() > 0))
				{
					// Get corresponding 3D point
					//cv::Point2f srcPt = device.keypointsFe1[device.stereoMatches[k][0].trainIdx].pt;
					
					// Solve Z from correspondence
					//double z = ((double)device.depth.at<short>(srcPt)) / 256.0;

					// Remove distant objects
					/*if ((z > 0.0 ) && (z < 50.0)) {*/
						// Solve 3D point
					cv::Point3f src3dpt;
					/*src3dpt.x = (float)(((double)srcPt.x - device.cx) * z / device.fx);
					src3dpt.y = (float)(((double)srcPt.y - device.cy) * z / device.fy);
					src3dpt.z = (float)z;*/
					device.stereoObjectPointsSrc.push_back(src3dpt);

					device.stereoKeypoints.push_back(device.keypointsFe1[device.stereoMatches[k][0].queryIdx]);
					device.stereoKeypointsSrc.push_back(device.keypointsFe2[device.stereoMatches[k][0].trainIdx]);

					device.stereoPoints.push_back(device.keypointsFe1[device.stereoMatches[k][0].queryIdx].pt);
					//keyframe->matchedPointsSrc.push_back(keypointsIr1[matches[k][0].trainIdx].pt);
					//device.stereoPointsSrc.push_back(srcPt);

					device.stereoDistances.push_back(device.stereoMatches[k][0].distance);
					//}
				}
			}
		}
		else {
			std::cout << "No relative matches found. " << std::endl;
		}
	}
	return 0;
}

int Tslam::imuPoseSolver() {
	bool isImuSettled = false;
	bool dropFirstTimestamp = false;
	int readCnt = 0;

	cv::Mat im = cv::Mat::zeros(100, 400, CV_8UC3);
	cv::putText(im, "IMU pose thread.", cv::Point(0, 30), cv::FONT_HERSHEY_PLAIN, 2, CV_RGB(255, 255, 255));
	cv::imshow("IMU", im);

	while (true) {
		char pressed = cv::waitKey(10);
		if (pressed == 27) break;

		// Let IMU settle first
		if (!isImuSettled) {
			if (!dropFirstTimestamp) {
				while (readCnt < 20) {
					if (processGyro(t265) && processAccel(t265)) {
						// Compute initial imu orientation from accelerometer. set yaw to zero.
						std::cout << t265.accel.x << " " << t265.accel.y << " " << t265.accel.z << std::endl;

						//MALI TO!
						float g = sqrtf(t265.accel.x * t265.accel.x + t265.accel.y * t265.accel.y + t265.accel.z*t265.accel.z);
						float a_x = t265.accel.x / g;
						float a_y = t265.accel.y / g;
						float a_z = t265.accel.z / g;
						std::cout << a_x << " " << a_y << " " << a_z << std::endl;

						float thetax = std::atan2f(a_y, a_z);
						float thetaz = std::atan2f(a_y, a_x);
						std::cout << thetax << " " << thetaz << std::endl;

						//glm::quat q(glm::vec3(-thetax -1.57079632679f, 0.0f, -thetaz));
						glm::quat q(glm::vec3(thetax, 0.0f, 0.0f));
						t265.ImuRotation = Quaternion(q.x, q.y, q.z, q.w);
						std::cout << std::fixed << t265.gyro.dt << " " << t265.accel.dt << std::endl;
						readCnt++;
					}
				}
				dropFirstTimestamp = true;
			}
			else {
				processGyro(t265);
				processAccel(t265);
				std::cout << "Settling" << std::endl;
				if (settleImu(t265))
					isImuSettled = true;
			}
		}
		else {
			if (!(processGyro(t265) && processAccel(t265)))
			{
				continue;
			}
			solveImuPose(t265);
			updateViewerImuPose(t265);
		}
	}
	return 0;
}

bool Tslam::settleImu(T265 &device) {
	solveImuPose(device);
	return true;
}

int Tslam::solveImuPose(T265 &device) {
	//float gyroMeasErrorX = GYRO_BIAS_X;
	//float gyroMeasErrorY = GYRO_BIAS_Y;
	//float gyroMeasErrorZ = GYRO_BIAS_Z;
	//float betax = 5.0f * gyroMeasErrorX;
	//float betay = 5.0f * gyroMeasErrorY;
	//float betax = 0.01f;
	float gyroMeasError = 3.14159265358979f * (5.0f / 180.0f); //5/180
	float betaw = 0.8f * gyroMeasError;

	float SEq_1 = device.ImuRotation.w;
	float SEq_2 = device.ImuRotation.x;
	float SEq_3 = device.ImuRotation.y;
	float SEq_4 = device.ImuRotation.z;

	float a_x = device.accel.x;
	float a_y = device.accel.y;
	float a_z = device.accel.z;

	float w_x = device.gyro.x;
	float w_y = device.gyro.y;
	float w_z = device.gyro.z;

	float norm;
	float SEqDot_omega_1, SEqDot_omega_2, SEqDot_omega_3, SEqDot_omega_4;
	float f_1, f_2, f_3;
	float J_11or24, J_12or23, J_13or22, J_14or21, J_32, J_33; // objective function Jacobian element
	float SEqHatDot_1, SEqHatDot_2, SEqHatDot_3, SEqHatDot_4;

	float halfSEq_1 = 0.5f * SEq_1;
	float halfSEq_2 = 0.5f * SEq_2;
	float halfSEq_3 = 0.5f * SEq_3;
	float halfSEq_4 = 0.5f * SEq_4;
	float twoSEq_1 = 2.0f * SEq_1;
	float twoSEq_2 = 2.0f * SEq_2;
	float twoSEq_3 = 2.0f * SEq_3;

	norm = sqrt(a_x * a_x + a_y * a_y + a_z * a_z);
	a_x /= norm;
	a_y /= norm;
	a_z /= norm;

	f_1 = twoSEq_2 * SEq_4 - twoSEq_1 * SEq_3 - a_x;
	f_2 = twoSEq_1 * SEq_2 + twoSEq_3 * SEq_4 - a_y;
	f_3 = 1.0f - twoSEq_2 * SEq_2 - twoSEq_3 * SEq_3 - a_z;

	J_11or24 = twoSEq_3; // J_11 negated in matrix multiplication
	J_12or23 = 2.0f * SEq_4;
	J_13or22 = twoSEq_1; // J_12 negated in matrix multiplication
	J_14or21 = twoSEq_2;
	J_32 = 2.0f * J_14or21; // negated in matrix multiplication
	J_33 = 2.0f * J_11or24; // negated in matrix multiplication

	// Compute the gradient (matrix multiplication)
	SEqHatDot_1 = J_14or21 * f_2 - J_11or24 * f_1;
	SEqHatDot_2 = J_12or23 * f_1 + J_13or22 * f_2 - J_32 * f_3;
	SEqHatDot_3 = J_12or23 * f_2 - J_33 * f_3 - J_13or22 * f_1;
	SEqHatDot_4 = J_14or21 * f_1 + J_11or24 * f_2;

	// Normalise the gradient
	norm = sqrt(SEqHatDot_1 * SEqHatDot_1 + SEqHatDot_2 * SEqHatDot_2 + SEqHatDot_3 * SEqHatDot_3 + SEqHatDot_4 * SEqHatDot_4);
	SEqHatDot_1 /= norm;
	SEqHatDot_2 /= norm;
	SEqHatDot_3 /= norm;
	SEqHatDot_4 /= norm;

	// Compute the quaternion derrivative measured by gyroscopes
	SEqDot_omega_1 = -halfSEq_2 * w_x - halfSEq_3 * w_y - halfSEq_4 * w_z;
	SEqDot_omega_2 = halfSEq_1 * w_x + halfSEq_3 * w_z - halfSEq_4 * w_y;
	SEqDot_omega_3 = halfSEq_1 * w_y - halfSEq_2 * w_z + halfSEq_4 * w_x;
	SEqDot_omega_4 = halfSEq_1 * w_z + halfSEq_2 * w_y - halfSEq_3 * w_x;

	// Compute then integrate the estimated quaternion derrivative
	SEq_1 += (SEqDot_omega_1 - (betaw * SEqHatDot_1)) * (float)device.gyro.dt;
	SEq_2 += (SEqDot_omega_2 - (betaw * SEqHatDot_2)) * (float)device.gyro.dt;
	SEq_3 += (SEqDot_omega_3 - (betaw * SEqHatDot_3)) * (float)device.gyro.dt;
	SEq_4 += (SEqDot_omega_4 - (betaw * SEqHatDot_4)) * (float)device.gyro.dt;

	// Normalise quaternion
	norm = sqrt(SEq_1 * SEq_1 + SEq_2 * SEq_2 + SEq_3 * SEq_3 + SEq_4 * SEq_4);
	SEq_1 /= norm;
	SEq_2 /= norm;
	SEq_3 /= norm;
	SEq_4 /= norm;

	device.ImuRotation.w = SEq_1;
	device.ImuRotation.x = SEq_2;
	device.ImuRotation.y = SEq_3;
	device.ImuRotation.z = SEq_4;

	/*device.ImuTranslation.x = device.ImuTranslation.x + device.ImuVelocity.x * (float)device.accel.dt;
	device.ImuTranslation.y = device.ImuTranslation.y + device.ImuVelocity.y * (float)device.accel.dt;
	device.ImuTranslation.z = device.ImuTranslation.z + device.ImuVelocity.z * (float)device.accel.dt;

	device.ImuVelocity.x = device.ImuVelocity.x + device.accel.x * (float)device.accel.dt;
	device.ImuVelocity.y = device.ImuVelocity.y + (device.accel.y + 9.81f) * (float)device.accel.dt;
	device.ImuVelocity.z = device.ImuVelocity.z + device.accel.z * (float)device.accel.dt;

	std::cout << device.accel.y << std::endl;*/

	//std::cout << SEq_1 << " " << SEq_2 << " " << SEq_3 << " " << SEq_4 << std::endl;
	//std::cout << device.gyro.x << " " << device.gyro.y << " " << device.gyro.z << std::endl;
	return 0;
}

int Tslam::visualizePose() {
	viewer->createWindow(800, 600, "campose");
	viewer->setCameraProjectionType(Viewer::ProjectionType::PERSPECTIVE);

	FileReader *objFile = new FileReader();
	float scale = 0.03f;
	objFile->readObj("monkey.obj", FileReader::ArrayFormat::VERTEX_NORMAL_TEXTURE, scale);
	FileReader *floorFile = new FileReader();
	floorFile->readObj("floor.obj", FileReader::ArrayFormat::VERTEX_NORMAL_TEXTURE, scale);
	FileReader *viewBoxFile = new FileReader();
	viewBoxFile->readObj("view.obj", FileReader::ArrayFormat::VERTEX_NORMAL_TEXTURE, scale);

	//Axis for Camera pose
	//cv::Mat texture = cv::imread("texture.png");
	CgObject *camAxis = new CgObject();
	camAxis->loadShader("myshader2.vert", "myshader2.frag");
	camAxis->loadData(objFile->vertexArray, objFile->indexArray, CgObject::ArrayFormat::VERTEX_NORMAL_TEXTURE);
	camAxis->loadTexture("monkey1.png");
	camAxis->setDrawMode(CgObject::Mode::TRIANGLES);
	camAxis->setLight();
	camAxis->objectIndex = (int)viewer->cgObject->size();
	viewer->cgObject->push_back(camAxis);

	// View box for Camera Pose
	CgObject *viewBox = new CgObject();
	viewBox->loadShader("edgeshader.vert", "edgeshader.frag");
	viewBox->loadData(viewBoxFile->vertexArray, viewBoxFile->indexArray, CgObject::ArrayFormat::VERTEX_NORMAL_TEXTURE);
	viewBox->loadTexture("default_texture.jpg");
	viewBox->setDrawMode(CgObject::Mode::CULLED_POINTS);
	viewBox->setLight();
	viewBox->setColor(cv::Scalar(0.0, 1.0, 0.0), 0.5);
	viewBox->objectIndex = (int)viewer->cgObject->size();
	viewer->cgObject->push_back(viewBox);

	//Axis for IMU pose
	//cv::Mat texture = cv::imread("texture.png");
	CgObject *imuAxis = new CgObject();
	imuAxis->loadShader("myshader2.vert", "myshader2.frag");
	imuAxis->loadData(objFile->vertexArray, objFile->indexArray, CgObject::ArrayFormat::VERTEX_NORMAL_TEXTURE);
	imuAxis->loadTexture("monkey2.png");
	imuAxis->setDrawMode(CgObject::Mode::TRIANGLES);
	imuAxis->setLight();
	imuAxis->objectIndex = (int)viewer->cgObject->size();
	viewer->cgObject->push_back(imuAxis);

	CgObject *imuViewBox = new CgObject();
	imuViewBox->loadShader("edgeshader.vert", "edgeshader.frag");
	imuViewBox->loadData(viewBoxFile->vertexArray, viewBoxFile->indexArray, CgObject::ArrayFormat::VERTEX_NORMAL_TEXTURE);
	imuViewBox->loadTexture("default_texture.jpg");
	imuViewBox->setDrawMode(CgObject::Mode::CULLED_POINTS);
	imuViewBox->setLight();
	imuViewBox->setColor(cv::Scalar(1.0, 0, 0.0));
	imuViewBox->objectIndex = (int)viewer->cgObject->size();
	viewer->cgObject->push_back(imuViewBox);

	CgObject *floor = new CgObject();
	floor->objectIndex = 4;
	floor->loadShader("myshader2.vert", "myshader2.frag");
	floor->loadData(floorFile->vertexArray, floorFile->indexArray, CgObject::ArrayFormat::VERTEX_NORMAL_TEXTURE);
	floor->loadTexture("floor.jpg");
	floor->setDrawMode(CgObject::Mode::TRIANGLES);
	floor->setLight();
	floor->objectIndex = (int)viewer->cgObject->size();
	viewer->cgObject->push_back(floor);
	viewer->cgObject->at(floor->objectIndex)->ty = -2.0f;

	viewer->run();
	viewer->close();

	return 0;
}

void Tslam::updateViewerImuPose(T265 &device) {
	if (viewer->isRunning) {
		float translationScale = 1.0f;

		// update imuaxis
		glm::quat imuAdjust = glm::quat(glm::vec3(-1.57079632679f, 0.0f, 0.0f));
		glm::quat imuQ(device.ImuRotation.w, device.ImuRotation.x, device.ImuRotation.y, device.ImuRotation.z);
		glm::quat imuQRot = imuAdjust * imuQ;
		glm::vec3 euler = glm::eulerAngles(imuQRot);

		viewer->cgObject->at(2)->qrot = imuQRot;
		//viewer->cgObject->at(2)->tx = -(float)t.at<double>(0) * translationScale;
		//viewer->cgObject->at(2)->ty = -(float)t.at<double>(1) * translationScale;
		//viewer->cgObject->at(2)->tz = (float)t.at<double>(2) *translationScale;

		viewer->cgObject->at(3)->qrot = imuQRot;
		//viewer->cgObject->at(3)->tx = -(float)t.at<double>(0) * translationScale;
		//viewer->cgObject->at(3)->ty = -(float)t.at<double>(1) * translationScale;
		//viewer->cgObject->at(3)->tz = (float)t.at<double>(2) *translationScale;

		cv::Mat im = cv::Mat::zeros(100, 300, CV_8UC3);
		im.setTo(cv::Scalar(50, 50, 50));
		overlayMatrixRot("imupose", im, Vector3(euler.x, euler.y, euler.z), device.ImuRotation);
	}
}

bool Tslam::processGyro(T265 &device) {

	device.gyro.lastTs = device.gyro.ts;
	rs2::frame gyroFrame;
	if (device.gyroQueue.poll_for_frame(&gyroFrame)) {
		rs2_vector gv = gyroFrame.as<rs2::motion_frame>().get_motion_data();

		device.gyro.ts = gyroFrame.get_timestamp();
		device.gyro.dt = (device.gyro.ts - device.gyro.lastTs) / 1000.0;
		device.gyro.x = gv.x;// -GYRO_BIAS_X;
		device.gyro.y = gv.y;// -GYRO_BIAS_Y;
		device.gyro.z = gv.z;// -GYRO_BIAS_Z;

		//std::cout << std::fixed
		//	<< device.gyro.ts << " " << device.gyro.lastTs << " "
		//	<< device.gyro.dt << ": ("
		//	<< device.gyro.x << ","
		//	<< device.gyro.y << ","
		//	<< device.gyro.z << " )"
		//	/*<< device.accel.dt << ": ("
		//	<< device.accel.x << " "
		//	<< device.accel.y << " "
		//	<< device.accel.z << ")"*/
		//	<< std::endl;

		return true;
	}
	else return false;
}

bool Tslam::processAccel(T265 &device) {
	device.accel.lastTs = device.accel.ts;
	rs2::frame accelFrame;

	if (device.accelQueue.poll_for_frame(&accelFrame)) {
		rs2_vector av = accelFrame.as<rs2::motion_frame>().get_motion_data();

		device.accel.ts = accelFrame.get_timestamp();
		device.accel.dt = (device.accel.ts - device.accel.lastTs) / 1000.0;
		device.accel.x = av.x;
		device.accel.y = av.y;
		device.accel.z = av.z;

		return true;
	}
	else return false;
}

/// Utilities
int Tslam::createDepthThresholdMask(T265 &device, float maxDepth) {
	//std::cout << device.depth32f.at<float>(320, 10) << std::endl;
	cv::threshold(device.depth32f, device.mask, (double)maxDepth, 255, cv::THRESH_BINARY_INV);
	device.mask.convertTo(device.mask, CV_8U);
	//std::cout << device.mask.type() << " " << CV_8U << " " << CV_8UC1 << std::endl;
	cv::imshow("thresh", device.mask);
	//std::cout << device.depth32f.at<float>(100, 100) << std::endl;
	return 0;
}

void Tslam::visualizeKeypoints(T265 &device, std::string windowNamePrefix) {
	cv::Mat imout1, imout2;
	cv::drawKeypoints(device.fisheye1, device.keypointsFe1, imout1, cv::Scalar(0, 255, 0), cv::DrawMatchesFlags::DEFAULT);
	cv::putText(imout1, "detected keypoints: " + parseDecimal((double)device.keypointsFe1.size(), 0), cv::Point(0, 10), cv::FONT_HERSHEY_PLAIN, 1, CV_RGB(255, 255, 255));
	cv::imshow(windowNamePrefix + "fisheye1", imout1);
	cv::drawKeypoints(device.fisheye2, device.keypointsFe2, imout2, cv::Scalar(0, 0, 255), cv::DrawMatchesFlags::DEFAULT);
	cv::putText(imout2, "detected keypoints: " + parseDecimal((double)device.keypointsFe2.size(), 0), cv::Point(0, 30), cv::FONT_HERSHEY_PLAIN, 1, CV_RGB(255, 255, 255));
	cv::imshow(windowNamePrefix + "fisheye2", imout2);
}

void Tslam::visualizeMatchedStereoPoints(T265 &device, std::string windowNamePrefix) {
	cv::Mat imout1, imout2;
	cv::drawKeypoints(device.fisheye1, device.stereoKeypoints, imout1, cv::Scalar(0, 255, 0), cv::DrawMatchesFlags::DEFAULT);
	cv::putText(imout1, "detected keypoints: " + parseDecimal((double)device.keypointsFe1.size(), 0), cv::Point(0, 10), cv::FONT_HERSHEY_PLAIN, 1, CV_RGB(255, 255, 255));
	cv::imshow(windowNamePrefix + "fisheye1", imout1);
	cv::drawKeypoints(device.fisheye2, device.stereoKeypointsSrc, imout2, cv::Scalar(0, 0, 255), cv::DrawMatchesFlags::DEFAULT);
	cv::putText(imout2, "matched keypoints: " + parseDecimal((double)device.stereoKeypointsSrc.size(), 0), cv::Point(0, 30), cv::FONT_HERSHEY_PLAIN, 1, CV_RGB(255, 255, 255));
	cv::imshow(windowNamePrefix + "fisheye2", imout2);
}

void Tslam::visualizeRelativeKeypoints(Keyframe *keyframe, cv::Mat ir1, std::string windowNamePrefix) {
	cv::Mat imout1, imout2;
	cv::drawKeypoints(keyframe->im, keyframe->keypoints, imout1, cv::Scalar(0, 255, 0), cv::DrawMatchesFlags::DEFAULT);
	cv::putText(imout1, "detected keypoints: " + parseDecimal((double)keyframe->keypoints.size(), 0), cv::Point(0, 10), cv::FONT_HERSHEY_PLAIN, 1, CV_RGB(255, 255, 255));
	cv::putText(imout1, "matched keypoints: " + parseDecimal((double)keyframe->matchedKeypoints.size(), 0), cv::Point(0, 30), cv::FONT_HERSHEY_PLAIN, 1, CV_RGB(255, 255, 255));
	cv::imshow(windowNamePrefix + "keyframe", imout1);
	cv::drawKeypoints(ir1, keyframe->matchedKeypointsSrc, imout2, cv::Scalar(0, 0, 255), cv::DrawMatchesFlags::DEFAULT);
	cv::putText(imout2, "matched keypoints: " + parseDecimal((double)keyframe->matchedKeypointsSrc.size(), 0), cv::Point(0, 30), cv::FONT_HERSHEY_PLAIN, 1, CV_RGB(255, 255, 255));
	cv::imshow(windowNamePrefix + "currentframe", imout2);
}

bool Tslam::isThisDevice(std::string serialNo, std::string queryNo) {
	if (std::strcmp(serialNo.c_str(), queryNo.c_str()) == 0) {
		return true;
	}
	else return false;
}

void Tslam::overlayMatrixRot(const char* windowName, cv::Mat& im, Vector3 euler, Quaternion q) {
	std::ostringstream message1, message2, message3;
	int precision = 3;
	message1 << std::fixed << this->parseDecimal(euler.x, precision) << " "
		<< this->parseDecimal(euler.y, precision) << " "
		<< this->parseDecimal(euler.z, precision);// << " " << this->parseDecimal(t.at<double>(0));
	message2 << std::fixed << this->parseDecimal(q.x, precision) << " "
		<< this->parseDecimal(q.y, precision) << " "
		<< this->parseDecimal(q.z, precision) << " "
		<< this->parseDecimal(q.w, precision);// << " " << this->parseDecimal(t.at<double>(1));
	//message3 << std::fixed << this->parseDecimal(R1.at<double>(2, 0)) << " " << this->parseDecimal(R1.at<double>(2, 1)) << " " << this->parseDecimal(R1.at<double>(2, 2)) << " " << this->parseDecimal(t.at<double>(2));
	cv::Mat overlay;
	double alpha = 0.3;
	im.copyTo(overlay);
	cv::rectangle(overlay, cv::Rect(0, 0, 400, 47), cv::Scalar(255, 255, 255), -1);
	cv::addWeighted(overlay, alpha, im, 1 - alpha, 0, im);
	//cv::rectangle(im, cv::Point(0, 0), cv::Point(256, 47), CV_RGB(255, 255, 255), CV_FILLED, cv::LINE_8, 0);
	cv::Scalar tc = CV_RGB(0, 0, 0);
	cv::putText(im, message1.str(), cv::Point(0, 10), cv::FONT_HERSHEY_PLAIN, 1, tc);
	cv::putText(im, message2.str(), cv::Point(0, 22), cv::FONT_HERSHEY_PLAIN, 1, tc);
	//cv::putText(im, message3.str(), cv::Point(0, 34), cv::FONT_HERSHEY_PLAIN, 1, tc);
	cv::imshow(windowName, im);
}

std::string Tslam::parseDecimal(double f) {
	std::stringstream string;
	if (f < 0) {
		//negative
		string.precision(5);
		string << std::fixed << f;
	}
	else {
		//positive or zero
		string.precision(5);
		string << std::fixed << "+" << f;
	}
	return string.str();
}

std::string Tslam::parseDecimal(double f, int precision) {
	std::stringstream string;
	if (f < 0) {
		//negative
		string.precision(precision);
		string << std::fixed << f;
	}
	else {
		//positive or zero
		string.precision(precision);
		string << std::fixed << "+" << f;
	}
	return string.str();
}


/// Tests
void Tslam::showDepthJet(std::string windowName, cv::Mat image, float maxDepth, bool shouldWait = true) {
	cv::Mat u_norm, u_gray, u_color;
	u_norm = image * 256.0f / maxDepth;
	u_norm.convertTo(u_gray, CV_8UC1);
	cv::applyColorMap(u_gray, u_color, cv::COLORMAP_JET);

	cv::imshow(windowName, u_color);
	if (shouldWait) cv::waitKey();
}

void Tslam::showDepthJet(std::string windowName, cv::Mat image, std::string message, float maxDepth, bool shouldWait = true) {
	cv::Mat u_norm, u_gray, u_color;
	u_norm = image * 256.0f / maxDepth;
	u_norm.convertTo(u_gray, CV_8UC1);
	cv::applyColorMap(u_gray, u_color, cv::COLORMAP_JET);
	cv::putText(u_color, message, cv::Point(10, 12), cv::FONT_HERSHEY_PLAIN, 1, CV_RGB(255, 255, 255));
	cv::imshow(windowName, u_color);
	if (shouldWait) cv::waitKey();
}

void Tslam::testStereo(std::string im1fn, std::string im2fn) {
	cv::Mat im1 = cv::imread(im1fn);
	cv::Mat im2 = cv::imread(im2fn);

	stereo = new Stereo();
	stereo->initializeFisheyeStereo(848, 800, 1, CV_8U, 6, 2.0f, 50.0f, 0.33f, 0.125f, 1, 1000);
	cv::Mat translationVector = cv::readOpticalFlow("translationVector.flo");
	cv::Mat calibrationVector = cv::readOpticalFlow("calibrationVector.flo");
	stereo->loadVectorFields(translationVector, calibrationVector);

	stereo->copyImagesToDevice(im1, im2);
	stereo->solveStereoForward();
	cv::Mat disparity = cv::Mat(800, 848, CV_32F);
	stereo->copyStereoToHost(disparity);
	cv::imshow("disparity", disparity / 50.0f);
	cv::imshow("im1", im1);
	cv::imshow("im2", im2);
	showDepthJet("color", disparity, 1, false);
	cv::waitKey();
}

int Tslam::saveT265Images(std::string filename, std::string folderOutput) {
	try {
		rs2::pipeline pipe;
		rs2::config cfg;

		cfg.enable_device_from_file(filename, false);
		cfg.enable_stream(RS2_STREAM_FISHEYE, 1, 848, 800, rs2_format::RS2_FORMAT_Y8, 30);
		cfg.enable_stream(RS2_STREAM_FISHEYE, 2, 848, 800, rs2_format::RS2_FORMAT_Y8, 30);
		//cfg.enable_stream(RS2_STREAM_ACCEL, RS2_FORMAT_MOTION_XYZ32F, 62);
		//cfg.enable_stream(RS2_STREAM_GYRO, RS2_FORMAT_MOTION_XYZ32F, 200);

		rs2::pipeline_profile profiles = pipe.start(cfg);
		rs2::device device = profiles.get_device();
		auto playback = device.as<rs2::playback>();
		//playback.set_playback_speed(0.1);
		playback.set_real_time(false);

		cv::Mat im = cv::Mat::zeros(100, 400, CV_8UC3);
		cv::putText(im, "Saving fisheye images.", cv::Point(0, 30), cv::FONT_HERSHEY_PLAIN, 2, CV_RGB(255, 255, 255));
		cv::imshow("Main", im);

		int ir1Cnt = 0;
		int ir2Cnt = 0;
		while (true)
		{
			char pressed = cv::waitKey(10);
			if (pressed == 27) break;
			rs2::frameset frameset;
			cv::Mat fisheye1, fisheye2;
			if (pipe.poll_for_frames(&frameset)) {
				auto fisheye1Data = frameset.get_fisheye_frame(1);
				auto fisheye2Data = frameset.get_fisheye_frame(2);

				fisheye1 = cv::Mat(cv::Size(848, 800), CV_8UC1, (void*)fisheye1Data.get_data(), cv::Mat::AUTO_STEP);
				fisheye2 = cv::Mat(cv::Size(848, 800), CV_8UC1, (void*)fisheye2Data.get_data(), cv::Mat::AUTO_STEP);

				cv::imwrite(folderOutput + "colored_0/data/im" + std::to_string(ir1Cnt) + ".png", fisheye1);
				ir1Cnt++;
				cv::imshow("ir1", fisheye1);
				cv::imwrite(folderOutput + "colored_1/data/im" + std::to_string(ir2Cnt) + ".png", fisheye2);
				ir2Cnt++;
				cv::imshow("ir2", fisheye1);
			}
		}
		pipe.stop();
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









// Trash codes
/*cv::Mat i1fixed = cv::imread("fs1.png", cv::IMREAD_GRAYSCALE);
cv::Mat i2fixed = cv::imread("fs2.png", cv::IMREAD_GRAYSCALE);
cv::equalizeHist(i1fixed, equi1);
cv::equalizeHist(i2fixed, equi2);*/
//cv::imshow("equi", equi1);

//cv::resize(t265.fisheye1, halfFisheye1, cv::Size(stereoWidth, stereoHeight));
//cv::resize(t265.fisheye2, halfFisheye2, cv::Size(stereoWidth, stereoHeight));

//std::cout << (int)halfFisheye1.at<uchar>(200, 200) << std::endl;

//stereotgv->solveStereoForward();

//stereo->solveStereoBackward();
//stereo->occlusionCheck(3.0f);


//std::cout << depth.at<float>(200, 200) << std::endl;
//std::cout << depthVis.cols << " " << depth.cols << std::endl;

/*cv::Mat planeSweepDepth = cv::Mat(stereoHeight, stereoWidth, CV_32F);
stereo->copyPlaneSweepToHost(planeSweepDepth);
cv::Mat planeSweepDepthVis;
planeSweepDepth.copyTo(planeSweepDepthVis, depthVisMask);
showDepthJet("psdepth", planeSweepDepthVis, 5.0f, false);*/