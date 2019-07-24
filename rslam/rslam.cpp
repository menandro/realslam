#include "rslam.h"

int Rslam::initialize(int width, int height, int fps) {
	viewer = new Viewer();

	this->width = width;
	this->height = height;
	this->fps = fps;

	// Gamma adjustment
	double gammaAdj = 0.5;
	lookUpTable = cv::Mat(1, 256, CV_8U);
	uchar* p = lookUpTable.ptr();
	for (int i = 0; i < 256; ++i) {
		p[i] = cv::saturate_cast<uchar>(pow(i / 255.0, gammaAdj) * 255.0);
	}

	try {
		ctx = new rs2::context();
		//pipe = new rs2::pipeline(*ctx);
		//pipelines = new std::vector<rs2::pipeline*>();
		auto dev = ctx->query_devices();
		
		for (auto&& devfound : dev) {
			const char * serialNo = devfound.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER);
			std::cout << "Found device: " << serialNo << std::endl;

			// Create pipeline for device0
			if (isThisDevice(serialNo, device0.serialNo)) {
				device0.pipe = new rs2::pipeline(*ctx);
			}
			if (isThisDevice(serialNo, device1.serialNo)) {
				device1.pipe = new rs2::pipeline(*ctx);
			}

			if (isThisDevice(serialNo, device0.serialNo) || isThisDevice(serialNo, device1.serialNo)) {
				std::cout << "Configuring " << serialNo << std::endl;
				// Turn off emitter
				
				auto depth_sensor = devfound.first<rs2::depth_sensor>();
				if (depth_sensor.supports(RS2_OPTION_EMITTER_ENABLED))
				{
					depth_sensor.set_option(RS2_OPTION_EMITTER_ENABLED, 0.0f); // Disable emitter
				}
				
				//rs2::config cfg;
				if (isThisDevice(serialNo, device0.serialNo)){// || (std::strcmp(serialNo, this->device1SN.c_str()) == 0)) {
					device0.cfg.enable_device(devfound.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER));
					device0.cfg.enable_stream(RS2_STREAM_DEPTH, this->width, this->height, rs2_format::RS2_FORMAT_Z16, this->fps);
					device0.cfg.enable_stream(RS2_STREAM_COLOR, this->width, this->height, RS2_FORMAT_BGR8, 60);
					device0.cfg.enable_stream(RS2_STREAM_INFRARED, 1, this->width, this->height, RS2_FORMAT_Y8, this->fps);
					device0.cfg.enable_stream(RS2_STREAM_INFRARED, 2, this->width, this->height, RS2_FORMAT_Y8, this->fps);
					device0.cfg.enable_stream(RS2_STREAM_ACCEL, RS2_FORMAT_MOTION_XYZ32F, 250);
					device0.cfg.enable_stream(RS2_STREAM_GYRO, RS2_FORMAT_MOTION_XYZ32F, 400);
				}

				if (isThisDevice(serialNo, device1.serialNo)) {// || (std::strcmp(serialNo, this->device1SN.c_str()) == 0)) {
					device1.cfg.enable_device(devfound.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER));
					device1.cfg.enable_stream(RS2_STREAM_DEPTH, this->width, this->height, rs2_format::RS2_FORMAT_Z16, this->fps);
					device1.cfg.enable_stream(RS2_STREAM_COLOR, this->width, this->height, RS2_FORMAT_BGR8, 60);
					device1.cfg.enable_stream(RS2_STREAM_INFRARED, 1, this->width, this->height, RS2_FORMAT_Y8, this->fps);
					device1.cfg.enable_stream(RS2_STREAM_INFRARED, 2, this->width, this->height, RS2_FORMAT_Y8, this->fps);
					//device1.cfg.enable_stream(RS2_STREAM_ACCEL, RS2_FORMAT_MOTION_XYZ32F, 250);
					//device1.cfg.enable_stream(RS2_STREAM_GYRO, RS2_FORMAT_MOTION_XYZ32F, 400);
				}
				//pipe->start(cfg);
				//pipelines.emplace_back(pipe);

				if (isThisDevice(serialNo, device0.serialNo)) {
					device0.pipe->start(device0.cfg);
					device0.depthScale = device0.pipe->get_active_profile().get_device()
						.query_sensors().front().as<rs2::depth_sensor>().get_depth_scale();
					device0.id = "device0";
					device0.isFound = true;
				}
				else if (isThisDevice(serialNo, device1.serialNo)) {
					device1.pipe->start(device1.cfg);
					device1.depthScale = device1.pipe->get_active_profile().get_device()
						.query_sensors().front().as<rs2::depth_sensor>().get_depth_scale();
					//device1.pipe = pipe;
					device1.id = "device1";
					device1.isFound = true;
				}
				std::cout << "Pipe created from: " << serialNo << std::endl;
			}
		}
		//std::cout << dev.size() << std::endl;
		//std::cout << "Found: " << dev[0].get_info(RS2_CAMERA_INFO_SERIAL_NUMBER) << std::endl;
		//std::cout << "Found: " << dev[1].get_info(RS2_CAMERA_INFO_SERIAL_NUMBER) << std::endl;
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

	alignToColor = rs2::align(RS2_STREAM_COLOR);

	/*camerapose = new CameraPose();
	double intrinsicData[9] = { fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0 };
	intrinsic = cv::Mat(3, 3, CV_64F, intrinsicData).clone();
	camerapose->initialize(intrinsic);*/

	// Utilities
	gyroDisp = cv::Mat::zeros(200, 600, CV_8UC3);
	accelDisp = cv::Mat::zeros(200, 600, CV_8UC3);

	// SLAM
	if (featMethod == SURF) {
		minHessian = 10000;
		surf = cv::cuda::SURF_CUDA(this->minHessian);
		surf.hessianThreshold = minHessian;
		matcher = cv::cuda::DescriptorMatcher::createBFMatcher(); //for surf
	}
	else if (featMethod == ORB){
		matcher = cv::cuda::DescriptorMatcher::createBFMatcher(cv::NORM_HAMMING); //for orb
		orb = cv::cuda::ORB::create(200, 2.0f, 1);// , 10, 0, 2, 0, 10);
		orb->setBlurForDescriptor(true);
		//orb = cv::cuda::ORB::create(200, 2.0f, 3, 10, 0, 2, 0, 10);
	}

	if (device0.isFound) initContainers(device0);
	if (device1.isFound) initContainers(device1);

	// Depth upsampling
	upsampling = new lup::Upsampling(32, 12, 32);
	int maxIter = 50;
	float beta = 9.0f;
	float gamma = 0.85f;
	float alpha0 = 17.0f;
	float alpha1 = 1.2f;
	float timeStepLambda = 1.0f;
	float lambdaTgvl2 = 5.0f;
	float maxDepth = 100.0f;
	this->maxDepth = maxDepth;
	upsampling->initialize(width, height, maxIter, beta, gamma, alpha0, alpha1, timeStepLambda, lambdaTgvl2, maxDepth);
	return EXIT_SUCCESS;
}

int Rslam::initializeFromFile(const char* filename0, const char* filenameImu) {
	viewer = new Viewer();
	setIntrinsics(device0, 320.729, 181.862, 321.902, 321.902);
	setIntrinsics(device1, 320.729, 181.862, 321.902, 321.902);
	this->width = 640;
	this->height = 360;
	this->fps = 90;
	this->featMethod = ORB;

	double gammaAdj = 0.5;
	lookUpTable = cv::Mat(1, 256, CV_8U);
	uchar* p = lookUpTable.ptr();
	for (int i = 0; i < 256; ++i) {
		p[i] = cv::saturate_cast<uchar>(pow(i / 255.0, gammaAdj) * 255.0);
	}

	try {
		device0.pipe = new rs2::pipeline();
		device0.cfg.enable_device_from_file(filename0, false);
		rs2::pipeline_profile profiled435i = device0.pipe->start(device0.cfg);
		rs2::device deviced435i = profiled435i.get_device();
		auto playbackd435i = deviced435i.as<rs2::playback>();
		playbackd435i.set_real_time(false);
		
		externalImu.pipe = new rs2::pipeline();
		externalImu.cfg.enable_device_from_file(filenameImu, false);
		externalImu.cfg.enable_stream(RS2_STREAM_ACCEL, RS2_FORMAT_MOTION_XYZ32F, 62);
		externalImu.cfg.enable_stream(RS2_STREAM_GYRO, RS2_FORMAT_MOTION_XYZ32F, 200);
		rs2::pipeline_profile profilet265 = externalImu.pipe->start(externalImu.cfg);
		rs2::device devicet265 = profilet265.get_device();
		auto playbackt265 = devicet265.as<rs2::playback>();
		playbackt265.set_real_time(false);

		device0.depthScale = device0.pipe->get_active_profile().get_device()
			.query_sensors().front().as<rs2::depth_sensor>().get_depth_scale();
		std::cout << "Depth scale: " << device0.depthScale << std::endl;

		std::cout << "Pipes started." << std::endl;
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

	// Utilities
	gyroDisp = cv::Mat::zeros(200, 600, CV_8UC3);
	accelDisp = cv::Mat::zeros(200, 600, CV_8UC3);

	// SLAM
	if (featMethod == SURF) {
		minHessian = 10000;
		surf = cv::cuda::SURF_CUDA(this->minHessian);
		surf.hessianThreshold = minHessian;
		matcher = cv::cuda::DescriptorMatcher::createBFMatcher(); //for surf
	}
	else if (featMethod == ORB) {
		matcher = cv::cuda::DescriptorMatcher::createBFMatcher(cv::NORM_HAMMING); //for orb
		orb = cv::cuda::ORB::create(2000, 2.0f, 2);// , 10, 0, 2, 0, 10);
		orb->setBlurForDescriptor(true);
		//orb = cv::cuda::ORB::create(200, 2.0f, 3, 10, 0, 2, 0, 10);
	}

	initContainers(device0);

	// Depth upsampling
	upsampling = new lup::Upsampling(32, 12, 32);
	int maxIter = 200;
	float beta = 9.0f;
	float gamma = 0.85f;
	float alpha0 = 17.0f;
	float alpha1 = 1.2f;
	float timeStepLambda = 1.0f;
	float lambdaTgvl2 = 5.0f;
	float maxDepth = 100.0f;
	this->maxDepth = maxDepth;
	upsampling->initialize(width, height, maxIter, beta, gamma, alpha0, alpha1, timeStepLambda, lambdaTgvl2, maxDepth);
	return EXIT_SUCCESS;
}

bool Rslam::isThisDevice(std::string serialNo, std::string queryNo) {
	if (std::strcmp(serialNo.c_str(), queryNo.c_str()) == 0) {
		return true;
	}
	else return false;
}

int Rslam::initialize(Settings settings, FeatureDetectionMethod featMethod, std::string device0SN, std::string device1SN) {
	this->featMethod = featMethod;
	//this->device0SN = device0SN;
	//this->device1SN = device1SN;
	this->device0.serialNo = device0SN;
	this->device1.serialNo = device1SN;
	return initialize(settings);
}

int Rslam::initialize(Settings settings) {
	int width, height, fps;
	double cx, cy, fx, fy;
	if (settings == T265) {
		fx = 614.122;
		fy = 614.365;
		cx = 427.388;
		cy = 238.478;
		width = 640;
		height = 480;
		fps = 60;
	}
	else if (settings == D435I_640_480_60) {
		fx = 614.122;
		fy = 614.365;
		cx = 323.388;
		cy = 238.478;
		width = 640;
		height = 480;
		fps = 60;
	}
	else if (settings == D435I_848_480_60) {
		fx = 614.122;
		fy = 614.365;
		cx = 427.388;
		cy = 238.478;
		width = 640;
		height = 480;
		fps = 60;
	}
	else if (settings == D435I_IR_640_360_90) {
		setIntrinsics(device0, 320.729, 181.862, 321.902, 321.902);
		setIntrinsics(device1, 320.729, 181.862, 321.902, 321.902);
		/*fx = 321.902;
		fy = 321.902;
		cx = 320.729;
		cy = 181.862;*/
		width = 640;
		height = 360;
		fps = 90;
	}
	else { //load default
		fx = 614.122;
		fy = 614.365;
		cx = 427.388;
		cy = 238.478;
		width = 640;
		height = 480;
		fps = 60;
	}
	return initialize(width, height, fps);
}

int Rslam::setIntrinsics(Device &device, double cx, double cy, double fx, double fy) {
	device.cx = cx;
	device.cy = cy;
	device.fx = fx;
	device.fy = fy;
	double intrinsicData[9] = { fx, 0, cx, 0, fy, cy, 0, 0 , 1 };
	device.intrinsic = cv::Mat(3, 3, CV_64F, intrinsicData).clone();
	device.distCoeffs = cv::Mat(4, 1, CV_64F);
	device.distCoeffs.at<double>(0) = 0;
	device.distCoeffs.at<double>(1) = 0;
	device.distCoeffs.at<double>(2) = 0;
	device.distCoeffs.at<double>(3) = 0;

	device.Rvec = cv::Mat::zeros(3, 1, CV_64F);
	device.t = cv::Mat::zeros(3, 1, CV_64F);

	

	return 0;
}

int Rslam::initContainers(Device &device) {
	device.depth = cv::Mat(this->height, this->width, CV_16U);
	device.mask = cv::Mat(this->height, this->width, CV_8U);
	device.depth32f = cv::Mat(this->height, this->width, CV_32F);
	device.depthVis = cv::Mat(this->height, this->width, CV_8UC3);
	device.color = cv::Mat(this->height, this->width, CV_8UC3);
	device.infrared1 = cv::Mat(this->height, this->width, CV_8UC1);
	device.infrared2 = cv::Mat(this->height, this->width, CV_8UC1);
	device.infrared132f = cv::Mat(this->height, this->width, CV_32F);
	device.infrared232f = cv::Mat(this->height, this->width, CV_32F);
	return 0;
}

// Thread calls
int Rslam::run() {
	//std::thread t1(&Rslam::visualizePose, this);
	//std::thread t2(&Rslam::poseSolver, this);
	//std::thread t2(&Rslam::poseSolverDefaultStereo, this);
	std::thread t0(&Rslam::fetchFrames, this);
	std::thread t1(&Rslam::imuPoseSolver, this);
	//std::thread t2(&Rslam::poseSolverDefaultStereoMulti, this);
	std::thread t2(&Rslam::cameraPoseSolver, this);
	std::thread t3(&Rslam::visualizePose, this);

	t0.join();
	t1.join();
	t2.join();
	t3.join();
}

int Rslam::runFromRecording() {
	
	std::thread t1(&Rslam::singleThread, this);
	std::thread t3(&Rslam::visualizePose, this);
	t1.join();
	t3.join();
}

// Main loop for pose estimation
int Rslam::singleThread() {
	cv::Mat im = cv::Mat::zeros(100, 400, CV_8UC3);
	cv::putText(im, "Main fetch thread.", cv::Point(0, 30), cv::FONT_HERSHEY_PLAIN, 2, CV_RGB(255, 255, 255));
	cv::imshow("Main", im);

	bool isImuSettled = false;
	bool dropFirstTimestamp = false;

	device0.currentKeyframe = new Keyframe();
	device0.keyframeExist = false;
	device0.currentKeyframe->R = cv::Mat::zeros(3, 1, CV_64F);
	device0.currentKeyframe->t = cv::Mat::zeros(3, 1, CV_64F);
	device0.currentKeyframe->currentRelativeR = cv::Mat::zeros(3, 1, CV_64F);
	device0.currentKeyframe->currentRelativeT = cv::Mat::zeros(3, 1, CV_64F);

	device1.currentKeyframe = new Keyframe();
	device1.keyframeExist = false;
	device1.currentKeyframe->R = cv::Mat::zeros(3, 1, CV_64F);
	device1.currentKeyframe->t = cv::Mat::zeros(3, 1, CV_64F);
	device1.currentKeyframe->currentRelativeR = cv::Mat::zeros(3, 1, CV_64F);
	device1.currentKeyframe->currentRelativeT = cv::Mat::zeros(3, 1, CV_64F);

	while (true) {
		char pressed = cv::waitKey(10);
		if (pressed == 27) break;

		// Poll framesets multi-camera (when any is available)
		//if (!mutex.try_lock()) continue;
		bool pollSuccess = (device0.pipe->poll_for_frames(&device0.frameset));
		//mutex.unlock();

		if (!pollSuccess) continue;

		auto gyroFrame = device0.frameset.first_or_default(RS2_STREAM_GYRO, RS2_FORMAT_MOTION_XYZ32F).as<rs2::motion_frame>();
		auto accelFrame = device0.frameset.first_or_default(RS2_STREAM_ACCEL, RS2_FORMAT_MOTION_XYZ32F).as<rs2::motion_frame>();
		auto depthData = device0.frameset.get_depth_frame();
		auto infrared1Data = device0.frameset.get_infrared_frame(1);
		//auto infrared2Data = device0.frameset.get_infrared_frame(2);
		device0.gyroQueue.enqueue(gyroFrame);
		device0.accelQueue.enqueue(accelFrame);
		/*device0.depthQueue.enqueue(depthData);
		device0.infrared1Queue.enqueue(infrared1Data);
		device0.infrared2Queue.enqueue(infrared2Data);*/
		device0.depth = cv::Mat(cv::Size(width, height), CV_16U, (void*)depthData.get_data(), cv::Mat::AUTO_STEP);
		device0.infrared1 = cv::Mat(cv::Size(width, height), CV_8UC1, (void*)infrared1Data.get_data(), cv::Mat::AUTO_STEP);
		// Adjust gamma
		//adjustGamma(device0);
		device0.infrared1.convertTo(device0.infrared132f, CV_32F, 1 / 256.0f);

		// Let IMU settle first
		if (!isImuSettled) {
			if (!dropFirstTimestamp) {
				if (processGyro(device0) && processAccel(device0)) {
					// Compute initial imu orientation from accelerometer. set yaw to zero.
					std::cout << device0.accel.x << " " << device0.accel.y << " " << device0.accel.z << std::endl;

					//MALI TO!
					float g = sqrtf(device0.accel.x * device0.accel.x + device0.accel.y * device0.accel.y + device0.accel.z*device0.accel.z);
					float a_x = device0.accel.x / g;
					float a_y = device0.accel.y / g;
					float a_z = device0.accel.z / g;

					float thetax = std::atan2f(a_y, a_z);
					float thetaz = std::atan2f(a_y, a_x);
					std::cout << thetax << " " << thetaz << std::endl;

					glm::quat q(glm::vec3(thetax, 0.0f, -thetaz));
					device0.ImuRotation = Quaternion(q.x, q.y, q.z, q.w);
					dropFirstTimestamp = true;
				}
			}
			else {
				processGyro(device0);
				processAccel(device0);
				std::cout << "Settling" << std::endl;
				if (settleImu(device0))
					isImuSettled = true;
			}
		}
		else {
			if (!(processGyro(device0) && processAccel(device0)))
			{
				continue;
			}
			solveImuPose(device0);
			updateViewerImuPose(device0);
		}

		if (pressed == 'r') {
			device0.keyframeExist = false; // change this to automatic keyframing
			device1.keyframeExist = false;
		}

		//if (!(processDepth(device0) && processIr(device0))) continue;
		upsampleDepth(device0);
		visualizeDepth(device0);
		createDepthThresholdMask(device0, 2.0f);

		detectAndComputeOrb(device0);
		//detectAndComputeOrb(device0.infrared1, device0.d_ir1, device0.keypointsIr1, device0.d_descriptorsIr1);
		//detectAndComputeOrb(device1.infrared1, device1.d_ir1, device1.keypointsIr1, device1.d_descriptorsIr1);

		// Match with keyframe
		matchAndPose(device0);
		//matchAndPose(device1);
		visualizeRelativeKeypoints(device0.currentKeyframe, device0.infrared1, "dev0");
		//visualizeRelativeKeypoints(device1.currentKeyframe, device1.infrared1, "dev1");

		Device viewDevice = device0;
		viewDevice.Rvec = viewDevice.currentKeyframe->currentRelativeR;
		viewDevice.t = viewDevice.currentKeyframe->currentRelativeT;
		this->Rvec = viewDevice.Rvec;
		this->t = viewDevice.t;
		cv::Mat im = cv::Mat::zeros(100, 300, CV_8UC3);
		im.setTo(cv::Scalar(50, 50, 50));
		overlayMatrix("pose", im, this->Rvec, this->t);

		updateViewerCameraPose(device0);
	}
	return 0;
}

int Rslam::fetchFrames() {
	cv::Mat im = cv::Mat::zeros(100, 400, CV_8UC3);
	cv::putText(im, "Main fetch thread.", cv::Point(0, 30), cv::FONT_HERSHEY_PLAIN, 2, CV_RGB(255, 255, 255));
	cv::imshow("Main", im);
	while (true) {
		char pressed = cv::waitKey(10);
		if (pressed == 27) break;

		// Poll framesets multi-camera (when any is available)
		//if (!mutex.try_lock()) continue;
		bool pollSuccess = (device0.pipe->poll_for_frames(&device0.frameset));
		//mutex.unlock();
		//device0.frameset = device0.pipe->wait_for_frames();
		if (!pollSuccess) continue;

		auto gyroFrame = device0.frameset.first_or_default(RS2_STREAM_GYRO, RS2_FORMAT_MOTION_XYZ32F).as<rs2::motion_frame>();
		auto accelFrame = device0.frameset.first_or_default(RS2_STREAM_ACCEL, RS2_FORMAT_MOTION_XYZ32F).as<rs2::motion_frame>();
		auto depthData = device0.frameset.get_depth_frame();
		auto infrared1Data = device0.frameset.get_infrared_frame(1);
		//auto infrared2Data = device0.frameset.get_infrared_frame(2);
		device0.gyroQueue.enqueue(gyroFrame);
		device0.accelQueue.enqueue(accelFrame);
		//device0.depthQueue.enqueue(depthData);
		//device0.infrared1Queue.enqueue(infrared1Data);
		//device0.infrared2Queue.enqueue(infrared2Data);
		/*rs2_vector gv = gyroFrame.as<rs2::motion_frame>().get_motion_data();
		device0.gyro.ts = gyroFrame.get_timestamp();
		device0.gyro.dt = (device0.gyro.ts - device0.gyro.lastTs) / 1000.0;
		device0.gyro.x = gv.x - GYRO_BIAS_X;
		device0.gyro.y = gv.y - GYRO_BIAS_Y;
		device0.gyro.z = gv.z - GYRO_BIAS_Z;

		rs2_vector av = accelFrame.as<rs2::motion_frame>().get_motion_data();
		device0.accel.ts = accelFrame.get_timestamp();
		device0.accel.dt = (device0.accel.ts - device0.accel.lastTs) / 1000.0;
		device0.accel.x = av.x;
		device0.accel.y = av.y;
		device0.accel.z = av.z;*/

		device0.depth = cv::Mat(cv::Size(width, height), CV_16U, (void*)depthData.get_data(), cv::Mat::AUTO_STEP);
		device0.infrared1 = cv::Mat(cv::Size(width, height), CV_8UC1, (void*)infrared1Data.get_data(), cv::Mat::AUTO_STEP);
		//adjustGamma(device0);
		device0.infrared1.convertTo(device0.infrared132f, CV_32F, 1 / 256.0f);
	}
	return 0;
}

int Rslam::imuPoseSolver() {
	bool isImuSettled = false;
	bool dropFirstTimestamp = false;
	
	while (true) {
		char pressed = cv::waitKey(10);
		if (pressed == 27) break;

		// Let IMU settle first
		if (!isImuSettled) {
			if (!dropFirstTimestamp) {
				if (processGyro(device0) && processAccel(device0)) {
					// Compute initial imu orientation from accelerometer. set yaw to zero.
					std::cout << device0.accel.x << " " << device0.accel.y << " " << device0.accel.z << std::endl;

					//MALI TO!
					float g = sqrtf(device0.accel.x * device0.accel.x + device0.accel.y * device0.accel.y + device0.accel.z*device0.accel.z);
					float a_x = device0.accel.x / g;
					float a_y = device0.accel.y / g;
					float a_z = device0.accel.z / g;

					float thetax = std::atan2f(a_y, a_z);
					float thetaz = std::atan2f(a_y, a_x);
					std::cout << thetax << " " << thetaz << std::endl;

					glm::quat q(glm::vec3(thetax, 0.0f, -thetaz));
					device0.ImuRotation = Quaternion(q.x, q.y, q.z, q.w);
					dropFirstTimestamp = true;
				}
			}
			else {
				processGyro(device0);
				processAccel(device0);
				std::cout << "Settling" << std::endl;
				if (settleImu(device0))
					isImuSettled = true;
			}
		}
		else {
			if (!(processGyro(device0) && processAccel(device0)))
			{
				continue;
			}
			solveImuPose(device0);
			updateViewerImuPose(device0);
		}
	}
	return 0;
}

int Rslam::cameraPoseSolver() {
	device0.currentKeyframe = new Keyframe();
	device0.keyframeExist = false;
	device0.currentKeyframe->R = cv::Mat::zeros(3, 1, CV_64F);
	device0.currentKeyframe->t = cv::Mat::zeros(3, 1, CV_64F);
	device0.currentKeyframe->currentRelativeR = cv::Mat::zeros(3, 1, CV_64F);
	device0.currentKeyframe->currentRelativeT = cv::Mat::zeros(3, 1, CV_64F);

	device1.currentKeyframe = new Keyframe();
	device1.keyframeExist = false;
	device1.currentKeyframe->R = cv::Mat::zeros(3, 1, CV_64F);
	device1.currentKeyframe->t = cv::Mat::zeros(3, 1, CV_64F);
	device1.currentKeyframe->currentRelativeR = cv::Mat::zeros(3, 1, CV_64F);
	device1.currentKeyframe->currentRelativeT = cv::Mat::zeros(3, 1, CV_64F);

	while (true) {
		char pressed = cv::waitKey(10);
		if (pressed == 27) break;
		if (pressed == 'r') {
			device0.keyframeExist = false; // change this to automatic keyframing
			device1.keyframeExist = false;
		}

		//if (!(processDepth(device0) && processIr(device0))) continue;
		upsampleDepth(device0);
		visualizeDepth(device0);
		createDepthThresholdMask(device0, 2.0f);

		detectAndComputeOrb(device0.infrared1, device0.d_ir1, device0.keypointsIr1, device0.d_descriptorsIr1);
		//detectAndComputeOrb(device1.infrared1, device1.d_ir1, device1.keypointsIr1, device1.d_descriptorsIr1);

		// Match with keyframe
		matchAndPose(device0);
		//matchAndPose(device1);
		visualizeRelativeKeypoints(device0.currentKeyframe, device0.infrared1, "dev0");
		//visualizeRelativeKeypoints(device1.currentKeyframe, device1.infrared1, "dev1");

		Device viewDevice = device0;
		viewDevice.Rvec = viewDevice.currentKeyframe->currentRelativeR;
		viewDevice.t = viewDevice.currentKeyframe->currentRelativeT;
		this->Rvec = viewDevice.Rvec;
		this->t = viewDevice.t;
		cv::Mat im = cv::Mat::zeros(100, 300, CV_8UC3);
		im.setTo(cv::Scalar(50, 50, 50));
		overlayMatrix("pose", im, this->Rvec, this->t);

		updateViewerCameraPose(device0);
	}
	return 0;
}

int Rslam::poseRefinement() {
	// Get current keyframe
	// Align depths with previous known keyframe
}


bool Rslam::settleImu(Device &device) {
	solveImuPose(device);
	return true;
}

int Rslam::solveImuPose(Device &device) {
	//float gyroMeasErrorX = GYRO_BIAS_X;
	//float gyroMeasErrorY = GYRO_BIAS_Y;
	//float gyroMeasErrorZ = GYRO_BIAS_Z;
	//float betax = 5.0f * gyroMeasErrorX;
	//float betay = 5.0f * gyroMeasErrorY;
	//float betax = 0.01f;
	float gyroMeasError = 3.14159265358979f * (5.0f / 180.0f);
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

int Rslam::matchAndPose(Device& device) {
	if (!device.keyframeExist) {
		device.currentKeyframe->im = device.infrared1.clone();
		device.currentKeyframe->d_im.upload(device.currentKeyframe->im);
		device.currentKeyframe->keypoints = device.keypointsIr1;
		device.currentKeyframe->d_descriptors = device.d_descriptorsIr1.clone();
		device.keyframeExist = true;
	}
	else {
		relativeMatchingDefaultStereo(device, device.currentKeyframe, device.infrared1);
		solveRelativePose(device, device.currentKeyframe);
	}
	return 0;
}

int Rslam::detectAndComputeOrb(cv::Mat im, cv::cuda::GpuMat &d_im, std::vector<cv::KeyPoint> &keypoints, cv::cuda::GpuMat &descriptors) {
	d_im.upload(im);
	orb->detectAndCompute(d_im, cv::cuda::GpuMat(), keypoints, descriptors);
	//orb->compute(d_ir1, keypoints, descriptors);
	return 0;
}

int Rslam::detectAndComputeOrb(Device& device) {
	device.d_ir1.upload(device.infrared1);
	device.d_mask.upload(device.mask);
	orb->detectAndCompute(device.d_ir1, device.d_mask, device.keypointsIr1, device.d_descriptorsIr1);
	//orb->compute(d_ir1, keypoints, descriptors);
	return 0;
}

int Rslam::relativeMatchingDefaultStereo(Device &device, Keyframe *keyframe, cv::Mat currentFrame) {
	if ((device.keypointsIr1.empty() || keyframe->keypoints.empty()) || (device.d_descriptorsIr1.cols <= 1) || (keyframe->d_descriptors.cols <= 1)) {
		std::cout << "No keypoints found." << std::endl;
	}
	else {
		matcher->knnMatch(keyframe->d_descriptors, device.d_descriptorsIr1, device.matches, 2);
		if (!device.matches.empty()) {
			//std::cout << "Matches: " << matches.size() << std::endl;
			keyframe->matchedKeypoints = std::vector< cv::KeyPoint >();
			keyframe->matchedKeypointsSrc = std::vector< cv::KeyPoint >();
			keyframe->matchedKeypoints.clear();
			keyframe->matchedKeypointsSrc.clear();

			keyframe->matchedPoints = std::vector<cv::Point2f>();
			keyframe->matchedPointsSrc = std::vector<cv::Point2f>();
			keyframe->matchedPoints.clear();
			keyframe->matchedPointsSrc.clear();

			keyframe->matchedDistances = std::vector< float >();
			keyframe->matchedDistances.clear();

			keyframe->objectPointsSrc = std::vector<cv::Point3f>();
			keyframe->objectPointsSrc.clear();

			for (int k = 0; k < (int)device.matches.size(); k++)
			{
				if ((device.matches[k][0].distance < 0.6*(device.matches[k][1].distance)) && ((int)device.matches[k].size() <= 2 && (int)device.matches[k].size() > 0))
				{
					// Get corresponding 3D point
					cv::Point2f srcPt = device.keypointsIr1[device.matches[k][0].trainIdx].pt;
					double z = ((double)device.depth.at<short>(srcPt)) / 256.0;

					// Remove distant objects
					/*if ((z > 0.0 ) && (z < 50.0)) {*/
						// Solve 3D point
						cv::Point3f src3dpt;
						src3dpt.x = (float)(((double)srcPt.x - device.cx) * z / device.fx);
						src3dpt.y = (float)(((double)srcPt.y - device.cy) * z / device.fy);
						src3dpt.z = (float)z;
						keyframe->objectPointsSrc.push_back(src3dpt);

						keyframe->matchedKeypoints.push_back(keyframe->keypoints[device.matches[k][0].queryIdx]);
						keyframe->matchedKeypointsSrc.push_back(device.keypointsIr1[device.matches[k][0].trainIdx]);

						keyframe->matchedPoints.push_back(keyframe->keypoints[device.matches[k][0].queryIdx].pt);
						//keyframe->matchedPointsSrc.push_back(keypointsIr1[matches[k][0].trainIdx].pt);
						keyframe->matchedPointsSrc.push_back(srcPt);

						keyframe->matchedDistances.push_back(device.matches[k][0].distance);
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

int Rslam::solveRelativePose(Device &device, Keyframe *keyframe) {
	//std::cout << this->intrinsic << std::endl;
	// Solve pose of keyframe wrt to current frame
	if (keyframe->matchedPoints.size() >= 15) {
		cv::solvePnPRansac(keyframe->objectPointsSrc, keyframe->matchedPoints, 
			device.intrinsic, device.distCoeffs, 
			keyframe->currentRelativeR, keyframe->currentRelativeT);
		//cv::solvePnP(keyframe->objectPointsSrc, keyframe->matchedPoints, this->intrinsic, this->distCoeffs, keyframe->R, keyframe->t, false);
	}
	// Convert R and t to the current frame
	return 0;
}

int Rslam::createDepthThresholdMask(Device& device, float maxDepth) {
	//std::cout << device.depth32f.at<float>(320, 10) << std::endl;
	cv::threshold(device.depth32f, device.mask, (double)maxDepth, 255, cv::THRESH_BINARY_INV);
	device.mask.convertTo(device.mask, CV_8U);
	//std::cout << device.mask.type() << " " << CV_8U << " " << CV_8UC1 << std::endl;
	cv::imshow("thresh", device.mask);
	//std::cout << device.depth32f.at<float>(100, 100) << std::endl;
	return 0;
}

// Process Frames
bool Rslam::processGyro(Device &device) {
	
	device.gyro.lastTs = device.gyro.ts;
	rs2::frame gyroFrame;
	if (device.gyroQueue.poll_for_frame(&gyroFrame)) {
		rs2_vector gv = gyroFrame.as<rs2::motion_frame>().get_motion_data();
		
		device.gyro.ts = gyroFrame.get_timestamp();
		device.gyro.dt = (device.gyro.ts - device.gyro.lastTs) / 1000.0;
		device.gyro.x = gv.x - GYRO_BIAS_X;
		device.gyro.y = gv.y - GYRO_BIAS_Y;
		device.gyro.z = gv.z - GYRO_BIAS_Z;

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

bool Rslam::processAccel(Device &device) {
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

bool Rslam::processDepth(Device &device) {
	rs2::frame depthData;
	if (device.depthQueue.poll_for_frame(&depthData)) {
		device.depth = cv::Mat(cv::Size(width, height), CV_16U, (void*)depthData.get_data(), cv::Mat::AUTO_STEP);
		//cv::imshow("just converted", device.depth);
		return true;
	}
	else return false;
}

bool Rslam::processIr(Device &device) {
	bool result1 = false;
	bool result2 = false;
	rs2::frame infrared1Data;
	if (device.infrared1Queue.poll_for_frame(&infrared1Data)) {
		device.infrared1 = cv::Mat(cv::Size(width, height), CV_8UC1, (void*)infrared1Data.get_data(), cv::Mat::AUTO_STEP);
		device.infrared1.convertTo(device.infrared132f, CV_32F, 1 / 256.0f);
		result1 = true;
	}	
	return result1;
	/*rs2::frame infrared2Data;
	if (device.infrared2Queue.poll_for_frame(&infrared2Data)) {
		device.infrared2 = cv::Mat(cv::Size(width, height), CV_8UC1, (void*)infrared2Data.get_data(), cv::Mat::AUTO_STEP);
		device.infrared2.convertTo(device.infrared232f, CV_32F, 1 / 256.0f);
		result2 = true;
	}
	return result1 & result2;*/
}

int Rslam::extractColor(Device &device) {
	auto colorData = device.frameset.get_color_frame();
	device.color = cv::Mat(cv::Size(width, height), CV_8UC3, (void*)colorData.get_data(), cv::Mat::AUTO_STEP);
	return 0;
}

int Rslam::upsampleDepth(Device &device) {
	device.depth.convertTo(device.depth32f, CV_32F, (double)device.depthScale);
	//std::cout << device.depth32f.at<float>(320, 160) << std::endl;

	upsampling->copyImagesToDevice(device.infrared132f, device.depth32f);
	upsampling->propagateColorOnly(10);
	//upsampling->solve();
	upsampling->copyImagesToHost(device.depth32f);

	device.depth32f = device.depth32f * this->maxDepth;
	//std::cout << device.depth32f.at<float>(320, 160) << std::endl;
	//device.depth32f.convertTo(device.depth, CV_16U, 1.0 / (double)device.depthScale);
	return 0;
}

int Rslam::adjustGamma(Device &device) {
	cv::Mat res = device.infrared1.clone();
	cv::LUT(device.infrared1, lookUpTable, device.infrared1);
	cv::imshow("gamma adjust", device.infrared1);
	return 0;
}


// Utilities
void Rslam::visualizeImu(Device &device) {
	std::ostringstream gyroValx, gyroValy, gyroValz;
	gyroValx << std::fixed << parseDecimal(device.gyro.x);
	gyroValy << std::fixed << parseDecimal(device.gyro.y);
	gyroValz << std::fixed << parseDecimal(device.gyro.z);
	//gyroDisp.setTo(cv::Scalar((gyro.x + 10) * 20, (gyro.y + 10) * 20, (gyro.z + 10) * 20));
	gyroDisp.setTo(cv::Scalar(50, 50, 50));
	cv::circle(gyroDisp, cv::Point(100, 100), (int)abs(10.0*device.gyro.x), cv::Scalar(0, 0, 255), -1);
	cv::circle(gyroDisp, cv::Point(300, 100), (int)abs(10.0*device.gyro.y), cv::Scalar(0, 255, 0), -1);
	cv::circle(gyroDisp, cv::Point(500, 100), (int)abs(10.0*device.gyro.z), cv::Scalar(255, 0, 0), -1);

	cv::putText(gyroDisp, gyroValx.str(), cv::Point(0, 10), cv::FONT_HERSHEY_PLAIN, 1, CV_RGB(255, 255, 255));
	cv::putText(gyroDisp, gyroValy.str(), cv::Point(200, 10), cv::FONT_HERSHEY_PLAIN, 1, CV_RGB(255, 255, 255));
	cv::putText(gyroDisp, gyroValz.str(), cv::Point(400, 10), cv::FONT_HERSHEY_PLAIN, 1, CV_RGB(255, 255, 255));
	cv::imshow("gyro", gyroDisp);

	std::ostringstream accelValx, accelValy, accelValz;
	accelValx << std::fixed << parseDecimal(device.accel.x);
	accelValy << std::fixed << parseDecimal(device.accel.y);
	accelValz << std::fixed << parseDecimal(device.accel.z);
	//gyroDisp.setTo(cv::Scalar((gyro.x + 10) * 20, (gyro.y + 10) * 20, (gyro.z + 10) * 20));
	accelDisp.setTo(cv::Scalar(50, 50, 50));
	cv::circle(accelDisp, cv::Point(100, 100), (int)abs(5.0*device.accel.x), cv::Scalar(0, 0, 255), -1);
	cv::circle(accelDisp, cv::Point(300, 100), (int)abs(5.0*device.accel.y), cv::Scalar(0, 255, 0), -1);
	cv::circle(accelDisp, cv::Point(500, 100), (int)abs(5.0*device.accel.z), cv::Scalar(255, 0, 0), -1);

	cv::putText(accelDisp, accelValx.str(), cv::Point(0, 10), cv::FONT_HERSHEY_PLAIN, 1, CV_RGB(255, 255, 255));
	cv::putText(accelDisp, accelValy.str(), cv::Point(200, 10), cv::FONT_HERSHEY_PLAIN, 1, CV_RGB(255, 255, 255));
	cv::putText(accelDisp, accelValz.str(), cv::Point(400, 10), cv::FONT_HERSHEY_PLAIN, 1, CV_RGB(255, 255, 255));
	cv::imshow("accel", accelDisp);
}

void Rslam::visualizePose() {
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
}

void Rslam::toEuler(Quaternion q, Vector3 &euler)
{
	// roll (x-axis rotation)
	float rollx, pitchy, yawz;
	float sinr_cosp = +2.0f * (q.w * q.x + q.y * q.z);
	float cosr_cosp = +1.0f - 2.0f * (q.x * q.x + q.y * q.y);
	rollx = atan2(sinr_cosp, cosr_cosp);

	// pitch (y-axis rotation)
	float sinp = 2.0f * (q.w * q.y - q.z * q.x);
	if (fabs(sinp) >= 1.0f)
		pitchy = copysign(3.14159f / 2.0f, sinp); // use 90 degrees if out of range
	else
		pitchy = asin(sinp);

	// yaw (z-axis rotation)
	float siny_cosp = 2.0f * (q.w * q.z + q.x * q.y);
	float cosy_cosp = 1.0f - 2.0f * (q.y * q.y + q.z * q.z);
	yawz = atan2(siny_cosp, cosy_cosp);

	euler.x = rollx;
	euler.y = pitchy;
	euler.z = yawz;
}

void Rslam::toQuaternion(Vector3 euler, Quaternion &q) // yaw (Z), pitch (Y), roll (X)
{
	// Abbreviations for the various angular functions
	double rollx = (double)euler.x;
	double pitchy = (double)euler.y;
	double yawz = (double)euler.z;

	double cy = cos(yawz * 0.5);
	double sy = sin(yawz * 0.5);

	double cp = cos(pitchy * 0.5);
	double sp = sin(pitchy * 0.5);

	double cr = cos(rollx * 0.5);
	double sr = sin(rollx * 0.5);

	q.w = (float)(cy * cp * cr + sy * sp * sr);
	q.x = (float)(cy * cp * sr - sy * sp * cr);
	q.y = (float)(sy * cp * sr + cy * sp * cr);
	q.z = (float)(sy * cp * cr - cy * sp * sr);
}

void Rslam::updateViewerCameraPose(Device &device) {
	if (viewer->isRunning) {
		float translationScale = 1.0f;
		//update camaxis

		glm::quat q(glm::vec3(-(float)Rvec.at<double>(0), -(float)Rvec.at<double>(1), (float)Rvec.at<double>(2)));
		viewer->cgObject->at(0)->qrot = q;

		viewer->cgObject->at(0)->tx = -(float)t.at<double>(0) * translationScale;
		viewer->cgObject->at(0)->ty = -(float)t.at<double>(1) * translationScale;
		viewer->cgObject->at(0)->tz = (float)t.at<double>(2) * translationScale;

		viewer->cgObject->at(1)->qrot = q;

		viewer->cgObject->at(1)->tx = -(float)t.at<double>(0) * translationScale;
		viewer->cgObject->at(1)->ty = -(float)t.at<double>(1) * translationScale;
		viewer->cgObject->at(1)->tz = (float)t.at<double>(2) * translationScale;
	}
}

void Rslam::updateViewerImuPose(Device &device) {
	if (viewer->isRunning) {
		float translationScale = 1.0f;

		// update imuaxis
		glm::quat imuAdjust = glm::quat(glm::vec3(-1.57079632679f, -1.57079632679f, 0.0f));
		glm::quat imuQ(device.ImuRotation.w, -device.ImuRotation.x, -device.ImuRotation.y, device.ImuRotation.z);
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

void Rslam::visualizeRelativeKeypoints(Keyframe *keyframe, cv::Mat ir1, std::string windowNamePrefix) {
	cv::Mat imout1, imout2;
	cv::drawKeypoints(keyframe->im, keyframe->keypoints, imout1, cv::Scalar(0, 255, 0), cv::DrawMatchesFlags::DEFAULT);
	cv::putText(imout1, "detected keypoints: " + parseDecimal((double)keyframe->keypoints.size(), 0), cv::Point(0, 10), cv::FONT_HERSHEY_PLAIN, 1, CV_RGB(255, 255, 255));
	cv::putText(imout1, "matched keypoints: " + parseDecimal((double)keyframe->matchedKeypoints.size(), 0), cv::Point(0, 30), cv::FONT_HERSHEY_PLAIN, 1, CV_RGB(255, 255, 255));
	cv::imshow(windowNamePrefix + "keyframe", imout1);
	cv::drawKeypoints(ir1, keyframe->matchedKeypointsSrc, imout2, cv::Scalar(0, 0, 255), cv::DrawMatchesFlags::DEFAULT);
	cv::putText(imout2, "matched keypoints: " + parseDecimal((double)keyframe->matchedKeypointsSrc.size(), 0), cv::Point(0, 30), cv::FONT_HERSHEY_PLAIN, 1, CV_RGB(255, 255, 255));
	cv::imshow(windowNamePrefix + "currentframe", imout2);
}

void Rslam::visualizeFps(double fps) {
	cv::Mat im = cv::Mat::zeros(400, 400, CV_8UC3);
	im.setTo(cv::Scalar(50, 50, 50));
	
	//cv::circle(im, cv::Point(200, 200), 120, cv::Scalar(50, 50, 50), -1);
	cv::circle(im, cv::Point(200, 200), (int)(std::min(fps*1.5, 120.0)), cv::Scalar(255, 255, 255), -1);
	cv::circle(im, cv::Point(200, 200), 90, cv::Scalar(0, 0, 255), 1);

	cv::putText(im, parseDecimal(fps, 1), cv::Point(0, 30), cv::FONT_HERSHEY_PLAIN, 2, CV_RGB(255, 255, 255));
	cv::imshow("fps", im);
}

void Rslam::visualizeColor(Device &device) {
	////auto colorData = frameset.get_color_frame();
	////color = cv::Mat(cv::Size(width, height), CV_8UC3, (void*)colorData.get_data(), cv::Mat::AUTO_STEP);
	cv::imshow(device.id + "color", device.color);
}

void Rslam::visualizeDepth(Device &device) {
	//auto depthData = device.frameset.get_depth_frame();
	/*auto depthVisData = colorizer.colorize(device.depth);
	device.depthVis = cv::Mat(cv::Size(width, height), CV_8UC3, (void*)depthVisData.get_data(), cv::Mat::AUTO_STEP);
	cv::imshow(device.id + "depth", device.depthVis);*/
	cv::Mat cm_img0;
	// Apply the colormap:
	cv::Mat depth8u;
	device.depth32f.convertTo(depth8u, CV_8UC1, 256.0f / 5.0f);
	cv::applyColorMap(depth8u, cm_img0, cv::COLORMAP_JET);
	// Show the result:
	cv::imshow("cm_img0", cm_img0);
}

void Rslam::overlayMatrix(const char* windowName, cv::Mat &im, cv::Mat R1, cv::Mat t) {
	std::ostringstream message1, message2, message3;
	int precision = 3;
	message1 << std::fixed << this->parseDecimal(R1.at<double>(0), precision) << " " 
		<< this->parseDecimal(R1.at<double>(1), precision) << " " 
		<< this->parseDecimal(R1.at<double>(2), precision);// << " " << this->parseDecimal(t.at<double>(0));
	message2 << std::fixed << this->parseDecimal(t.at<double>(0), precision) << " " 
		<< this->parseDecimal(t.at<double>(1), precision) << " " 
		<< this->parseDecimal(t.at<double>(2), precision);// << " " << this->parseDecimal(t.at<double>(1));
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

void Rslam::overlayMatrixRot(const char* windowName, cv::Mat& im, Vector3 euler, Quaternion q) {
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


// Tools
std::string Rslam::parseDecimal(double f) {
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

std::string Rslam::parseDecimal(double f, int precision) {
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

int Rslam::saveImu(const char* filename0, const char* filenameImu, std::string outputFolder) {
	try {
		std::map<int, int> counters;
		std::map<int, int> frameNumber;
		std::map<int, std::string> stream_names;
		std::mutex mutex;

		std::ofstream accelFile(outputFolder + "accel.csv");
		accelFile << "x,y,z,ts,frame" << std::endl;
		std::ofstream gyroFile(outputFolder + "gyro.csv");
		gyroFile << "x,y,z,ts,frame" << std::endl;

		auto callback = [&](const rs2::frame& frame)
		{
			//std::lock_guard<std::mutex> lock(mutex);
			if (rs2::frameset fs = frame.as<rs2::frameset>())
			{
				for (const rs2::frame& f : fs) {
					// Save depth, infrared1, infrared2
					//std::cout << f.get_profile().unique_id() << ":" << f.get_frame_number() << " ";
				}
			}
			else
			{
				// Save gyro, accel
				if (frame.get_profile().stream_type() == RS2_STREAM_GYRO 
					&& frame.get_profile().format() == RS2_FORMAT_MOTION_XYZ32F)
				{
					auto gyroFrame = frame.as<rs2::motion_frame>();
					rs2_vector gv = gyroFrame.get_motion_data();
					gyroFile << std::fixed 
						<< gv.x << "," 
						<< gv.y << "," 
						<< gv.z << "," 
						<< gyroFrame.get_timestamp() << "," 
						<< frame.get_frame_number() 
						<< std::endl;
					std::cout << frame.get_profile().unique_id() << ":" << frame.get_frame_number() << " ";
				}
				else if (frame.get_profile().stream_type() == RS2_STREAM_ACCEL 
					&& frame.get_profile().format() == RS2_FORMAT_MOTION_XYZ32F) {
					auto accelFrame = frame.as<rs2::motion_frame>();
					rs2_vector gv = accelFrame.get_motion_data();
					accelFile << std::fixed
						<< gv.x << ","
						<< gv.y << ","
						<< gv.z << ","
						<< accelFrame.get_timestamp() << ","
						<< frame.get_frame_number()
						<< std::endl;
					std::cout << frame.get_profile().unique_id() << ":" << frame.get_frame_number() << " ";
				}
				//std::cout << frame.get_profile().unique_id() << ":" << frame.get_frame_number() << " ";
			}
			std::cout << "\r";
		};

		rs2::pipeline pipe;
		rs2::config cfg;

		cfg.enable_device_from_file(filename0, false);
		/*d435i.cfg.enable_stream(RS2_STREAM_DEPTH, 640, 360, rs2_format::RS2_FORMAT_Z16, 90);
		d435i.cfg.enable_stream(RS2_STREAM_COLOR, 640, 360, RS2_FORMAT_BGR8, 60);
		d435i.cfg.enable_stream(RS2_STREAM_INFRARED, 1, 640, 360, RS2_FORMAT_Y8, 90);
		d435i.cfg.enable_stream(RS2_STREAM_INFRARED, 2, 640, 360, RS2_FORMAT_Y8, 90);*/
		cfg.enable_stream(RS2_STREAM_ACCEL, RS2_FORMAT_MOTION_XYZ32F, 250);
		cfg.enable_stream(RS2_STREAM_GYRO, RS2_FORMAT_MOTION_XYZ32F, 400);

		rs2::pipeline_profile profiles = pipe.start(cfg, callback);
		rs2::device device = profiles.get_device();
		auto playback = device.as<rs2::playback>();
		//playback.set_playback_speed(0.1);
		//playback.set_real_time(false);
		
		cv::Mat im = cv::Mat::zeros(100, 400, CV_8UC3);
		cv::putText(im, "Main fetch thread.", cv::Point(0, 30), cv::FONT_HERSHEY_PLAIN, 2, CV_RGB(255, 255, 255));
		cv::imshow("Main", im);
		while (true) {
			char pressed = cv::waitKey(10);
			if (pressed == 27) break;
		}

		gyroFile.close();
		accelFile.close();
		/*cv::Mat im = cv::Mat::zeros(100, 400, CV_8UC3);
		cv::putText(im, "Main fetch thread.", cv::Point(0, 30), cv::FONT_HERSHEY_PLAIN, 2, CV_RGB(255, 255, 255));
		cv::imshow("Main", im);
		cv::waitKey();*/

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

int Rslam::saveExternalImu(const char* filename0, const char* filenameImu, std::string outputFolder) {
	try {
		std::map<int, int> counters;
		std::map<int, int> frameNumber;
		std::map<int, std::string> stream_names;
		std::mutex mutex;

		std::ofstream accelFile(outputFolder + "externalaccel.csv");
		accelFile << "x,y,z,ts,frame" << std::endl;
		std::ofstream gyroFile(outputFolder + "externalgyro.csv");
		gyroFile << "x,y,z,ts,frame" << std::endl;

		auto callback = [&](const rs2::frame& frame)
		{
			//std::lock_guard<std::mutex> lock(mutex);
			if (auto motion = frame.as<rs2::motion_frame>())
			{
				// Save gyro, accel
				if (motion.get_profile().stream_type() == RS2_STREAM_GYRO
					&& motion.get_profile().format() == RS2_FORMAT_MOTION_XYZ32F)
				{
					auto gyroFrame = motion.as<rs2::motion_frame>();
					rs2_vector gv = gyroFrame.get_motion_data();
					gyroFile << std::fixed
						<< gv.x << ","
						<< gv.y << ","
						<< gv.z << ","
						<< gyroFrame.get_timestamp() << ","
						<< motion.get_frame_number()
						<< std::endl;
					std::cout << motion.get_profile().unique_id() << ":" << motion.get_frame_number() << " ";
				}
				else if (motion.get_profile().stream_type() == RS2_STREAM_ACCEL
					&& motion.get_profile().format() == RS2_FORMAT_MOTION_XYZ32F) {
					auto accelFrame = motion.as<rs2::motion_frame>();
					rs2_vector gv = accelFrame.get_motion_data();
					accelFile << std::fixed
						<< gv.x << ","
						<< gv.y << ","
						<< gv.z << ","
						<< accelFrame.get_timestamp() << ","
						<< motion.get_frame_number()
						<< std::endl;
					std::cout << motion.get_profile().unique_id() << ":" << motion.get_frame_number() << " ";
				}
			}
			std::cout << "\r";
		};

		rs2::pipeline pipe;
		rs2::config cfg;

		cfg.enable_device_from_file(filenameImu, false);
		cfg.enable_stream(RS2_STREAM_ACCEL, RS2_FORMAT_MOTION_XYZ32F, 62);
		cfg.enable_stream(RS2_STREAM_GYRO, RS2_FORMAT_MOTION_XYZ32F, 200);

		rs2::pipeline_profile profiles = pipe.start(cfg, callback);
		rs2::device device = profiles.get_device();
		auto playback = device.as<rs2::playback>();
		//playback.set_playback_speed(0.1);
		//playback.set_real_time(false);

		cv::Mat im = cv::Mat::zeros(100, 400, CV_8UC3);
		cv::putText(im, "Main fetch thread.", cv::Point(0, 30), cv::FONT_HERSHEY_PLAIN, 2, CV_RGB(255, 255, 255));
		cv::imshow("Main", im);
		while (true) {
			char pressed = cv::waitKey(10);
			if (pressed == 27) break;
		}

		gyroFile.close();
		accelFile.close();
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

int Rslam::getSynchronization(const char* filename0, const char* filenameImu, std::string outputFolder) {
	Device d435i;
	Device t265;
	try {
		std::cout << "Loading file..." << std::endl;
		d435i.pipe = new rs2::pipeline();
		d435i.cfg.enable_device_from_file(filename0, false);
		rs2::pipeline_profile profiled435i = d435i.pipe->start(d435i.cfg);
		rs2::device deviced435i = profiled435i.get_device();
		auto playbackd435i = deviced435i.as<rs2::playback>();
		playbackd435i.set_real_time(true);
		//playbackd435i.set_playback_speed(0.005f);

		t265.pipe = new rs2::pipeline();
		t265.cfg.enable_device_from_file(filenameImu, false);
		t265.cfg.enable_stream(RS2_STREAM_ACCEL, RS2_FORMAT_MOTION_XYZ32F, 62);
		t265.cfg.enable_stream(RS2_STREAM_GYRO, RS2_FORMAT_MOTION_XYZ32F, 200);
		rs2::pipeline_profile profilet265 = t265.pipe->start(t265.cfg);
		rs2::device devicet265 = profilet265.get_device();
		auto playbackt265 = devicet265.as<rs2::playback>();
		playbackt265.set_real_time(true);
		//playbackt265.set_playback_speed(0.005f);

		rs2::colorizer color_map;

		cv::Mat im = cv::Mat::zeros(100, 400, CV_8UC3);
		cv::putText(im, "Main fetch thread.", cv::Point(0, 30), cv::FONT_HERSHEY_PLAIN, 2, CV_RGB(255, 255, 255));
		cv::imshow("Main", im);
		int depthCnt = 0;
		int ir1Cnt = 0;
		int ir2Cnt = 0;
		int accelCnt = 0;
		int gyroCnt = 0;
		int externalGyroCnt = 0;
		int externalAccelCnt = 0;

		// Open gyro and accel files x, y, z, timestamp
		std::ofstream syncFile(outputFolder + "sync.csv");
		syncFile << "depth,image,accel,gyro,externalAccel,externalGyro" << std::endl;

		while (true)
		{
			char pressed = cv::waitKey(10);
			if (pressed == 27) break;
			double depthTs, imageTs, accelTs, gyroTs, externalAccelTs, externalGyroTs;

			if (d435i.pipe->poll_for_frames(&d435i.frameset)) {
				imageTs = d435i.frameset.get_infrared_frame(1).get_timestamp();
				depthTs = d435i.frameset.get_depth_frame().get_timestamp();

				// Save gyro with timestamp
				auto gyroFrame = d435i.frameset.first_or_default(RS2_STREAM_GYRO,
					RS2_FORMAT_MOTION_XYZ32F).as<rs2::motion_frame>();
				rs2_vector gv = gyroFrame.get_motion_data();
				gyroTs = gyroFrame.get_timestamp();

				// Save accel with timestamp
				auto accelFrame = d435i.frameset.first_or_default(RS2_STREAM_ACCEL,
					RS2_FORMAT_MOTION_XYZ32F).as<rs2::motion_frame>();
				rs2_vector av = accelFrame.get_motion_data();
				accelTs = accelFrame.get_timestamp();

				//for (const rs2::frame& f : d435i.frameset) {
				//	// Save depth as 16-bit
				//	if (f.is<rs2::depth_frame>()) {
				//		
				//	}
				//	if (f.is<rs2::depth_frame>()) {
				//		
				//	}
				//	// Save infrared1
				//	// Save infrared2
				//	// Save gyro and timestamp
				//	// Save accel and timestamp
				//	//std::cout << f.get_profile().unique_id() << ":" << f.get_frame_number() << " || ";
				//}
				//std::cout << std::endl;
				accelCnt++;
			}


			if (t265.pipe->poll_for_frames(&t265.frameset)) {
				// Save gyro with timestamp
				auto gyroFrame = t265.frameset.first_or_default(RS2_STREAM_GYRO,
					RS2_FORMAT_MOTION_XYZ32F).as<rs2::motion_frame>();
				rs2_vector gv = gyroFrame.get_motion_data();
				externalGyroTs = gyroFrame.get_timestamp();

				// Save accel with timestamp
				auto accelFrame = t265.frameset.first_or_default(RS2_STREAM_ACCEL,
					RS2_FORMAT_MOTION_XYZ32F).as<rs2::motion_frame>();
				rs2_vector av = accelFrame.get_motion_data();
				externalAccelTs = accelFrame.get_timestamp();

				//t265.pipe->wait_for_frames();
					//std::cout << "t265:: ";
					//// Get matching timestamps here
					////auto accelFrame = t265.frameset.first_or_default(RS2_STREAM_ACCEL, RS2_FORMAT_MOTION_XYZ32F).as<rs2::motion_frame>();
					////std::cout << std::fixed << accelFrame.get_timestamp() << std::endl;

					//// Save all frames here
					//for (const rs2::frame& f : t265.frameset) {
					//	// Save depth as 16-bit
					//	// Save infrared1
					//	// Save infrared2
					//	// Save gyro and timestamp
					//	// Save accel and timestamp
					//	std::cout << f.get_profile().unique_id() << ":" << f.get_frame_number() << " || ";
					//}
					//std::cout << std::endl;
				externalAccelCnt++;
			}
			syncFile << std::fixed
				<< depthTs << ","
				<< imageTs << ","
				<< accelTs << ","
				<< gyroTs << ","
				<< externalAccelTs << ","
				<< externalGyroTs << ","
				<< std::endl;

			std::cout << "\r";
			std::cout << "frames: " << accelCnt << " " << externalAccelCnt;
		}
		syncFile.close();
		d435i.pipe->stop();
		t265.pipe->stop();
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

int Rslam::saveAllDepthAndInfrared(const char* filename0, const char* filenameImu, std::string outputFolder) {
	Device d435i;
	Device t265;
	try {
		d435i.pipe = new rs2::pipeline();
		d435i.cfg.enable_device_from_file(filename0, false);
		rs2::pipeline_profile profiled435i = d435i.pipe->start(d435i.cfg);
		rs2::device deviced435i = profiled435i.get_device();
		auto playbackd435i = deviced435i.as<rs2::playback>();
		playbackd435i.set_real_time(false);
		//playbackd435i.set_playback_speed(0.005f);

		t265.pipe = new rs2::pipeline();
		t265.cfg.enable_device_from_file(filenameImu, false);
		t265.cfg.enable_stream(RS2_STREAM_ACCEL, RS2_FORMAT_MOTION_XYZ32F, 62);
		t265.cfg.enable_stream(RS2_STREAM_GYRO, RS2_FORMAT_MOTION_XYZ32F, 200);
		rs2::pipeline_profile profilet265 = t265.pipe->start(t265.cfg);
		rs2::device devicet265 = profilet265.get_device();
		auto playbackt265 = devicet265.as<rs2::playback>();
		playbackt265.set_real_time(false);
		//playbackt265.set_playback_speed(0.005f);

		rs2::colorizer color_map;

		cv::Mat im = cv::Mat::zeros(100, 400, CV_8UC3);
		cv::putText(im, "Main fetch thread.", cv::Point(0, 30), cv::FONT_HERSHEY_PLAIN, 2, CV_RGB(255, 255, 255));
		cv::imshow("Main", im);
		int depthCnt = 0;
		int ir1Cnt = 0;
		int ir2Cnt = 0;
		int accelCnt = 0;
		int gyroCnt = 0;
		int externalGyroCnt = 0;
		int externalAccelCnt = 0;

		// Open gyro and accel files x, y, z, timestamp
		std::ofstream imageTimestampFile(outputFolder + "infrared1/timestamp.csv");
		std::ofstream depthTimestampFile(outputFolder + "depth/timestamp.csv");

		while (true)
		{
			char pressed = cv::waitKey(10);
			if (pressed == 27) break;

			if (d435i.pipe->poll_for_frames(&d435i.frameset)) {
				// Save image frames here
				d435i.depth = cv::Mat(cv::Size(640, 360), CV_16U, (void*)d435i.frameset.get_depth_frame().get_data(), cv::Mat::AUTO_STEP);
				cv::imwrite(outputFolder + "depth/" + std::to_string(depthCnt) + ".png", d435i.depth);
				d435i.depth = cv::Mat(cv::Size(640, 360), CV_8UC3, (void*)d435i.frameset.get_depth_frame().apply_filter(color_map).get_data(), cv::Mat::AUTO_STEP);
				cv::imwrite(outputFolder + "depthvis/" + std::to_string(depthCnt) + ".png", d435i.depth);
				depthCnt++;

				d435i.infrared1 = cv::Mat(cv::Size(640, 360), CV_8UC1, (void*)d435i.frameset.get_infrared_frame(1).get_data(), cv::Mat::AUTO_STEP);
				cv::imwrite(outputFolder + "infrared1/" + std::to_string(ir1Cnt) + ".png", d435i.infrared1);
				ir1Cnt++;
				cv::imshow("ir1", d435i.infrared1);

				double imageTs = d435i.frameset.get_infrared_frame(1).get_timestamp();
				double depthTs = d435i.frameset.get_depth_frame().get_timestamp();

				imageTimestampFile << std::fixed << imageTs << std::endl;
				depthTimestampFile << std::fixed << depthTs << std::endl;

				std::cout << "\r";
				std::cout << "frames: " << ir1Cnt;
			}
		}
		imageTimestampFile.close();
		depthTimestampFile.close();
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
