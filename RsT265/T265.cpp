#include "T265.h"

T265::T265() {

}

int T265::initialize(const char* serialNumber, float stereoScaling) {
	this->stereoScaling = stereoScaling;
	return this->initialize(serialNumber);
}

int T265::initialize(const char* serialNumber) {
	//initStereoTVL1();
	stereoWidth = (int)(width / stereoScaling);
	stereoHeight = (int)(height / stereoScaling);
	double intrinsicData[9] = { 285.722, 0, 420.135, 0, 286.759, 403.394, 0, 0 , 1 };
	intrinsic = cv::Mat(3, 3, CV_64F, intrinsicData).clone();
	distCoeffs = cv::Mat(4, 1, CV_64F);
	distCoeffs.at<double>(0) = -0.00659769;
	distCoeffs.at<double>(1) = 0.0473251;
	distCoeffs.at<double>(2) = -0.0458264;
	distCoeffs.at<double>(3) = 0.00897725;

	initStereoTGVL1();

	try {
		ctx = new rs2::context();
		auto dev = ctx->query_devices();

		for (auto&& devfound : dev) {
			const char* serialNo = devfound.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER);
			std::cout << "Found device: " << serialNo << std::endl;
			std::vector<rs2::sensor> sensors = devfound.query_sensors();
			sensors[0].set_option(RS2_OPTION_ENABLE_AUTO_EXPOSURE, 0);
			sensors[0].set_option(RS2_OPTION_EXPOSURE, 12000);
			sensors[0].set_option(RS2_OPTION_GAIN, 1);

			// Create pipeline for device0
			if (isThisDevice(serialNo, serialNumber)) {
				this->serialNo = serialNumber;
				this->pipe = new rs2::pipeline(*ctx);

				std::cout << "Configuring " << serialNo << std::endl;
				this->cfg.enable_stream(RS2_STREAM_FISHEYE, 1, 848, 800, rs2_format::RS2_FORMAT_Y8, 30);
				this->cfg.enable_stream(RS2_STREAM_FISHEYE, 2, 848, 800, rs2_format::RS2_FORMAT_Y8, 30);
				this->cfg.enable_stream(RS2_STREAM_ACCEL, RS2_FORMAT_MOTION_XYZ32F, 62);
				this->cfg.enable_stream(RS2_STREAM_GYRO, RS2_FORMAT_MOTION_XYZ32F, 200);

				auto prof = this->pipe->start(this->cfg);
				// Disable auto-exposure


				this->isFound = true;
				std::cout << "Pipe created from: " << serialNo << std::endl;
			}
		}
	}
	catch (const rs2::error & e)
	{
		std::cerr << "RealSense error calling " << e.get_failed_function() << "(" << e.get_failed_args() << "):\n    " << e.what() << std::endl;
		return EXIT_FAILURE;
	}
	catch (const std::exception & e)
	{
		std::cerr << e.what() << std::endl;
		return EXIT_FAILURE;
	}

	this->fisheye1 = cv::Mat(this->height, this->width, CV_8UC1);
	this->fisheye2 = cv::Mat(this->height, this->width, CV_8UC1);
	this->fisheye132f = cv::Mat(this->height, this->width, CV_32F);
	this->fisheye232f = cv::Mat(this->height, this->width, CV_32F);

	// Create fisheye mask
	//this->fisheyeMask = cv::Mat::zeros(cv::Size(this->width, this->height), CV_8UC1);
	//cv::circle(this->fisheyeMask, cv::Point(this->stereoWidth, this->stereoHeight), this->stereoWidth - 40, cv::Scalar(256.0f), -1);

	return 0;
}

int T265::run() {
	std::thread t0(&T265::fetchFrames, this);
	std::thread t1(&T265::imuPoseSolver, this);
	isFetchFramesRunning = true;
	isImuPoseSolverRunning = true;
	std::thread t2(&T265::stopThreadCheck, this);
	std::cout << "All threads started" << std::endl;

	t0.detach();
	t1.detach();
	t2.detach();
	//this->pipe->stop();
	
	return 0;
}

int T265::stopThreadCheck() {
	while (this->isFetchFramesRunning || this->isImuPoseSolverRunning) {
		std::this_thread::sleep_for(std::chrono::seconds(1)); // So we don't check flags every time
	}
	this->stop();
	std::cout << "Pipe stopped." << std::endl;
	return 0;
}

int T265::stop() {
	this->pipe->stop();
	return 0;
}

int T265::fetchFrames() {
	cv::Mat im = cv::Mat::zeros(100, 400, CV_8UC3);
	cv::putText(im, "T265 fetch thread.", cv::Point(0, 30), cv::FONT_HERSHEY_PLAIN, 2, CV_RGB(255, 255, 255));
	cv::imshow("T265 Device", im);
	while (true) {
		char pressed = cv::waitKey(10);
		if (pressed == 27) break;

		// Poll framesets multi-camera (when any is available)
		//if (!mutex.try_lock()) continue;
		bool pollSuccess = (this->pipe->poll_for_frames(&this->frameset));
		//mutex.unlock();
		//device0.frameset = device0.pipe->wait_for_frames();
		if (!pollSuccess) continue;

		auto gyroFrame = this->frameset.first_or_default(RS2_STREAM_GYRO, RS2_FORMAT_MOTION_XYZ32F).as<rs2::motion_frame>();
		auto accelFrame = this->frameset.first_or_default(RS2_STREAM_ACCEL, RS2_FORMAT_MOTION_XYZ32F).as<rs2::motion_frame>();
		auto fisheye1Data = this->frameset.get_fisheye_frame(1);
		auto fisheye2Data = this->frameset.get_fisheye_frame(2);
		this->gyroQueue.enqueue(gyroFrame);
		this->accelQueue.enqueue(accelFrame);

		this->fisheye1 = cv::Mat(cv::Size(this->width, this->height), 
			CV_8UC1, (void*)fisheye1Data.get_data(), cv::Mat::AUTO_STEP);
		this->fisheye2 = cv::Mat(cv::Size(this->width, this->height), 
			CV_8UC1, (void*)fisheye2Data.get_data(), cv::Mat::AUTO_STEP);
		
		if (stereoScaling == 2.0f) {
			cv::resize(this->fisheye1, this->image, cv::Size(stereoWidth, stereoHeight));
		}
		else this->image = this->fisheye1;

		solveStereoTGVL1();

		cv::Mat equi1, rgb;
		cv::equalizeHist(this->fisheye1, equi1);
		cv::cvtColor(equi1, this->fisheye1texture, cv::COLOR_GRAY2RGB);
	}
	isFetchFramesRunning = false;
	std::cout << "Fetch stopped: " << isFetchFramesRunning << std::endl;
	return 0;
}

int T265::imuPoseSolver() {
	bool isImuSettled = false;
	bool dropFirstTimestamp = false;
	int readCnt = 0;

	cv::Mat im = cv::Mat::zeros(100, 400, CV_8UC3);
	cv::putText(im, "IMU-to-Pose Conversion thread.", cv::Point(0, 30), cv::FONT_HERSHEY_PLAIN, 2, CV_RGB(255, 255, 255));
	cv::imshow("T265 IMU-to-Pose Conversion Thread", im);

	while (true) {
		char pressed = cv::waitKey(10);
		if (pressed == 27) break;

		// Let IMU settle first
		if (!isImuSettled) {
			if (!dropFirstTimestamp) {
				while (readCnt < 20) {
					if (processGyro() && processAccel()) {
						// Compute initial imu orientation from accelerometer. set yaw to zero.
						//std::cout << this->accel.x << " " << this->accel.y << " " << this->accel.z << std::endl;

						//MALI TO!
						float g = sqrtf(this->accel.x * this->accel.x + 
							this->accel.y * this->accel.y + 
							this->accel.z * this->accel.z);
						float a_x = this->accel.x / g;
						float a_y = this->accel.y / g;
						float a_z = this->accel.z / g;
						//std::cout << a_x << " " << a_y << " " << a_z << std::endl;

						float thetax = std::atan2f(a_y, a_z);
						float thetaz = std::atan2f(a_y, a_x);
						//std::cout << thetax << " " << thetaz << std::endl;

						//glm::quat q(glm::vec3(-thetax -1.57079632679f, 0.0f, -thetaz));
						glm::quat q(glm::vec3(thetax, 0.0f, 0.0f));
						this->ImuRotation = Quaternion(q.x, q.y, q.z, q.w);
						//std::cout << std::fixed << this->gyro.dt << " " << this->accel.dt << std::endl;
						readCnt++;
					}
				}
				dropFirstTimestamp = true;
			}
			else {
				processGyro();
				processAccel();
				std::cout << "Settling" << std::endl;
				if (settleImu())
					isImuSettled = true;
			}
		}
		else {
			if (!(processGyro() && processAccel()))
			{
				continue;
			}
			solveImuPose(); // Actual Function
			//updateViewerImuPose(t265);
		}
	}
	isImuPoseSolverRunning = false;
	std::cout << "Imu stopped: " << isImuPoseSolverRunning << std::endl;
	return 0;
}

bool T265::settleImu() {
	solveImuPose();
	return true;
}

int T265::solveImuPose() {
	//float gyroMeasErrorX = GYRO_BIAS_X;
	//float gyroMeasErrorY = GYRO_BIAS_Y;
	//float gyroMeasErrorZ = GYRO_BIAS_Z;
	//float betax = 5.0f * gyroMeasErrorX;
	//float betay = 5.0f * gyroMeasErrorY;
	//float betax = 0.01f;
	float gyroMeasError = 3.14159265358979f * (5.0f / 180.0f); //5/180
	float betaw = 0.8f * gyroMeasError;

	float SEq_1 = this->ImuRotation.w;
	float SEq_2 = this->ImuRotation.x;
	float SEq_3 = this->ImuRotation.y;
	float SEq_4 = this->ImuRotation.z;

	float a_x = this->accel.x;
	float a_y = this->accel.y;
	float a_z = this->accel.z;

	float w_x = this->gyro.x;
	float w_y = this->gyro.y;
	float w_z = this->gyro.z;

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
	SEq_1 += (SEqDot_omega_1 - (betaw * SEqHatDot_1)) * (float)this->gyro.dt;
	SEq_2 += (SEqDot_omega_2 - (betaw * SEqHatDot_2)) * (float)this->gyro.dt;
	SEq_3 += (SEqDot_omega_3 - (betaw * SEqHatDot_3)) * (float)this->gyro.dt;
	SEq_4 += (SEqDot_omega_4 - (betaw * SEqHatDot_4)) * (float)this->gyro.dt;

	// Normalise quaternion
	norm = sqrt(SEq_1 * SEq_1 + SEq_2 * SEq_2 + SEq_3 * SEq_3 + SEq_4 * SEq_4);
	SEq_1 /= norm;
	SEq_2 /= norm;
	SEq_3 /= norm;
	SEq_4 /= norm;

	this->ImuRotation.w = SEq_1;
	this->ImuRotation.x = SEq_2;
	this->ImuRotation.y = SEq_3;
	this->ImuRotation.z = SEq_4;

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

bool T265::processGyro() {

	this->gyro.lastTs = this->gyro.ts;
	rs2::frame gyroFrame;
	if (this->gyroQueue.poll_for_frame(&gyroFrame)) {
		rs2_vector gv = gyroFrame.as<rs2::motion_frame>().get_motion_data();

		this->gyro.ts = gyroFrame.get_timestamp();
		this->gyro.dt = (this->gyro.ts - this->gyro.lastTs) / 1000.0;
		this->gyro.x = gv.x;// -GYRO_BIAS_X;
		this->gyro.y = gv.y;// -GYRO_BIAS_Y;
		this->gyro.z = gv.z;// -GYRO_BIAS_Z;

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

bool T265::processAccel() {
	this->accel.lastTs = this->accel.ts;
	rs2::frame accelFrame;

	if (this->accelQueue.poll_for_frame(&accelFrame)) {
		rs2_vector av = accelFrame.as<rs2::motion_frame>().get_motion_data();

		this->accel.ts = accelFrame.get_timestamp();
		this->accel.dt = (this->accel.ts - this->accel.lastTs) / 1000.0;
		this->accel.x = av.x;
		this->accel.y = av.y;
		this->accel.z = av.z;

		return true;
	}
	else return false;
}


bool T265::isThisDevice(std::string serialNo, std::string queryNo) {
	if (std::strcmp(serialNo.c_str(), queryNo.c_str()) == 0) {
		return true;
	}
	else return false;
}