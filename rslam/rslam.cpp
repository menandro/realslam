#include "rslam.h"

int Rslam::initialize(int width, int height, int fps) {
	viewer = new Viewer();

	this->width = width;
	this->height = height;
	this->fps = fps;

	try {
		ctx = new rs2::context();
		//pipe = new rs2::pipeline(*ctx);
		//pipelines = new std::vector<rs2::pipeline*>();
		auto dev = ctx->query_devices();
		for (auto&& devfound : dev) {
			rs2::pipeline * pipe = new rs2::pipeline(*ctx);
			rs2::config cfg;

			std::cout << "Found devices: " << devfound.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER) << std::endl;
			const char * serialNo = devfound.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER);
			if ((std::strcmp(serialNo, this->device0SN.c_str()) == 0) || (std::strcmp(serialNo, this->device1SN.c_str()) == 0)) {
				std::cout << "Configuring " << serialNo << std::endl;
				// Turn off emitter
				
				auto depth_sensor = devfound.first<rs2::depth_sensor>();
				if (depth_sensor.supports(RS2_OPTION_EMITTER_ENABLED))
				{
					depth_sensor.set_option(RS2_OPTION_EMITTER_ENABLED, 0.0f); // Disable emitter
				}
				
				rs2::config cfg;
				cfg.enable_device(devfound.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER));
				cfg.enable_stream(RS2_STREAM_DEPTH, this->width, this->height, rs2_format::RS2_FORMAT_Z16, this->fps);
				cfg.enable_stream(RS2_STREAM_COLOR, this->width, this->height, RS2_FORMAT_BGR8, 60);
				cfg.enable_stream(RS2_STREAM_INFRARED, 1, this->width, this->height, RS2_FORMAT_Y8, this->fps);
				cfg.enable_stream(RS2_STREAM_INFRARED, 2, this->width, this->height, RS2_FORMAT_Y8, this->fps);
				if ((std::strcmp(serialNo, this->device0SN.c_str()) == 0)){// || (std::strcmp(serialNo, this->device1SN.c_str()) == 0)) {
					cfg.enable_stream(RS2_STREAM_ACCEL, RS2_FORMAT_MOTION_XYZ32F, 250);
					cfg.enable_stream(RS2_STREAM_GYRO, RS2_FORMAT_MOTION_XYZ32F, 400);
				}
				pipe->start(cfg);
				pipelines.emplace_back(pipe);

				if ((std::strcmp(serialNo, this->device0SN.c_str()) == 0)) {
					device0.pipe = pipe;
					device0.id = "device0";
					device0.isFound = true;
				}
				else if ((std::strcmp(serialNo, this->device1SN.c_str()) == 0)) {
					device1.pipe = pipe;
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
		//orb = cv::cuda::ORB::create(200, 2.0f, 3, 10, 0, 2, 0, 10);
	}

	if (device0.isFound) initContainers(device0);
	if (device0.isFound) initContainers(device1);

	// Depth upsampling
	upsampling = new lup::Upsampling(32, 12, 32);
	int maxIter = 50;
	float beta = 9.0f;
	float gamma = 0.85f;
	float alpha0 = 17.0f;
	float alpha1 = 1.2f;
	float timeStepLambda = 1.0f;
	float lambdaTgvl2 = 5.0f;
	float maxDepth = 10.0f;
	this->maxDepth = maxDepth;
	upsampling->initialize(width, height, maxIter, beta, gamma, alpha0, alpha1, timeStepLambda, lambdaTgvl2, maxDepth);
	return EXIT_SUCCESS;
}

int Rslam::initialize(Settings settings, FeatureDetectionMethod featMethod, std::string device0SN, std::string device1SN) {
	this->featMethod = featMethod;
	this->device0SN = device0SN;
	this->device1SN = device1SN;
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
	device.depth = cv::Mat(this->height, this->width, CV_16S);
	device.depth32f = cv::Mat(this->height, this->width, CV_32F);
	device.depthVis = cv::Mat(this->height, this->width, CV_8UC3);
	device.color = cv::Mat(this->height, this->width, CV_8UC3);
	device.infrared1 = cv::Mat(this->height, this->width, CV_8UC1);
	device.infrared2 = cv::Mat(this->height, this->width, CV_8UC1);
	device.infrared132f = cv::Mat(this->height, this->width, CV_32F);
	device.infrared232f = cv::Mat(this->height, this->width, CV_32F);
	return 0;
}

// Record feed from all available sensors
int Rslam::recordAll() {
	rs2::context context;
	std::vector<rs2::pipeline> pipelines;
	// Start a streaming pipe per each connected device
	auto tt = context.query_devices();
	std::cout << "Size " << tt.size() << std::endl;
	/*rs2::device d435i;
	rs2::device t265;*/
	std::clock_t start;
//	double duration;
	// Start pipes as recorders
	for (auto&& dev : context.query_devices())
	{
		rs2::pipeline pipe(context);
		rs2::config cfg;
		cfg.enable_device(dev.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER));
		std::string filename = dev.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER);
		filename.append(".bag");
		cfg.enable_record_to_file(filename);
		if (dev.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER) == "843112071357") {
			// d435 device, enable depth, 2IR, rgb, imu
			cfg.disable_all_streams();
			cfg.enable_stream(RS2_STREAM_DEPTH, 640, 360, rs2_format::RS2_FORMAT_Z16, 90);
			cfg.enable_stream(RS2_STREAM_INFRARED, 1, 640, 360, rs2_format::RS2_FORMAT_Y8, 90);
			cfg.enable_stream(RS2_STREAM_INFRARED, 2, 640, 360, rs2_format::RS2_FORMAT_Y8, 90);
			cfg.enable_stream(RS2_STREAM_COLOR, 640, 360, RS2_FORMAT_BGR8, 60);
			cfg.enable_stream(RS2_STREAM_ACCEL, RS2_FORMAT_MOTION_XYZ32F, 250);
			cfg.enable_stream(RS2_STREAM_GYRO, RS2_FORMAT_MOTION_XYZ32F, 400);
		}
		else {
			// t265 device, imu
			//cfg.enable_stream(RS2_STREAM_ACCEL, RS2_FORMAT_MOTION_XYZ32F, 250);
			//cfg.enable_stream(RS2_STREAM_GYRO, RS2_FORMAT_MOTION_XYZ32F, 400);
		}
		pipe.start(cfg);
		pipelines.emplace_back(pipe);
	}
	std::cout << "Recording..." << std::endl;
	std::cout << "Press g: stop recording." << std::endl;
	cv::Mat something = cv::imread("recording.png");
	cv::imshow("test", something);
	start = std::clock();
	while (true) {
		char pressed = cv::waitKey(10);
		//if (pressed == 27) break; //press escape

		//if (pressed == 'y'){
		//	std::cout << "Recording... press g to stop." << std::endl;
		//	// Set all pipelines as recorder
		//	for (int k = 0; k < pipelines.size(); k++) {
		//		rs2::device device = pipelines[k].get_active_profile().get_device();
		//		if (!device.as<rs2::recorder>()) {
		//			pipelines[k].stop();
		//			r
		//		}
		//	}
		//}
		if ((pressed == 27) || (pressed == 'g')) {// || ((std::clock() - start) / (double)CLOCKS_PER_SEC > 10.0)) {
			std::cout << "Saving recording to file... ";
			for (int k = 0; k < pipelines.size(); k++) {
				pipelines[k].stop();
			}
			std::cout << "DONE." << std::endl;
			break;
		}
	}
	return 0;
}

int Rslam::playback(const char* serialNumber) {
	try{
		rs2::pipeline pipe;
		rs2::config cfg;
		cfg.enable_device_from_file(serialNumber);
		pipe.start(cfg);
		std::cout << "Pipe started" << std::endl;
		const auto window_name = "Display Image";
		cv::namedWindow(window_name, cv::WINDOW_AUTOSIZE);
		rs2::colorizer color_map;

		while (cv::waitKey(1) < 0 && cv::getWindowProperty(window_name, cv::WND_PROP_AUTOSIZE) >= 0)
		{
			rs2::frameset data = pipe.wait_for_frames(); // Wait for next set of frames from the camera
			//std::cout << data.size() << std::endl;
			rs2::frame depth = data.get_depth_frame().apply_filter(color_map);

			// Query frame size (width and height)
			const int w = depth.as<rs2::video_frame>().get_width();
			const int h = depth.as<rs2::video_frame>().get_height();

			// Create OpenCV matrix of size (w,h) from the colorized depth data
			cv::Mat depthimage(cv::Size(w, h), CV_8UC3, (void*)depth.get_data(), cv::Mat::AUTO_STEP);

			// Update the window with new data
			cv::imshow(window_name, depthimage);
		}
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
}

// Thread calls
int Rslam::run() {
	//std::thread t1(&Rslam::visualizePose, this);
	//std::thread t2(&Rslam::poseSolver, this);
	//std::thread t2(&Rslam::poseSolverDefaultStereo, this);
	std::thread t2(&Rslam::poseSolverDefaultStereoMulti, this);
	std::thread t3(&Rslam::visualizePose, this);

	//t1.join();
	t2.join();
	t3.join();
}

// Main loop for pose estimation
int Rslam::poseSolverDefaultStereoMulti() {
//	double last_ts[RS2_STREAM_COUNT];
//	double dt[RS2_STREAM_COUNT];
	std::clock_t start;

	double timer = 0.0;
	double fps, oldFps = 0.0;

	cv::Mat prevInfrared1 = cv::Mat::zeros(cv::Size(this->width, this->height), CV_8UC1);
	
	// Placeholder TODO: move to per device
	device0.currentKeyframe = new Keyframe();
	device0.keyframeExist = false;
	device0.currentKeyframe->R = cv::Mat::zeros(3, 1, CV_64F);
	device0.currentKeyframe->t = cv::Mat::zeros(3, 1, CV_64F);
	device0.imuKeyframeExist = false;
	
	device1.currentKeyframe = new Keyframe();
	device1.keyframeExist = false;
	device1.currentKeyframe->R = cv::Mat::zeros(3, 1, CV_64F);
	device1.currentKeyframe->t = cv::Mat::zeros(3, 1, CV_64F);

	bool firstFewFramesDropped = false;

	while (true) {
		char pressed = cv::waitKey(10);
		if (pressed == 27) break;
		if (pressed == 'r') {
			device0.keyframeExist = false; // change this to automatic keyframing
			device1.keyframeExist = false;
			device0.imuKeyframeExist = false;
		}

		// Poll framesets multi-camera (when any is available)
		/*bool pollSuccess = (device0.pipe->poll_for_frames(&device0.frameset) | 
			device1.pipe->poll_for_frames(&device1.frameset));*/
		bool pollSuccess = (device0.pipe->poll_for_frames(&device0.frameset));
		if (!pollSuccess) continue;

		start = std::clock();

		if (!firstFewFramesDropped) {
			extractGyroAndAccel(device0); // dump first frame 
			firstFewFramesDropped = true;
		}
		else {
			solveImuPose(device0);
		}
		//visualizeImu();

		solveCameraPose();
		
		updateViewerPose(device0);

		fps = 1 / ((std::clock() - start) / (double)CLOCKS_PER_SEC);
		oldFps = (fps + oldFps) / 2.0; // Running average
		visualizeFps(oldFps);
	}
	return 0;
}

int Rslam::solveImuPose(Device &device) {
	if (!device.imuKeyframeExist) {
		Quaternion initialQ;
		toQuaternion(Vector3(-(3.14159265358979f / 2.0f), 0.0f, 0.0f), initialQ);
		device.ImuRotation = initialQ;
		device.imuKeyframeExist = true;
	}
	else {
		extractGyroAndAccel(device);

		float gyroMeasError = 3.14159265358979f * (5.0f / 180.0f);
		float beta = sqrtf(3.0f / 4.0f) * gyroMeasError;
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
		SEq_1 += (SEqDot_omega_1 - (beta * SEqHatDot_1)) * (float)device.gyro.dt;
		SEq_2 += (SEqDot_omega_2 - (beta * SEqHatDot_2)) * (float)device.gyro.dt;
		SEq_3 += (SEqDot_omega_3 - (beta * SEqHatDot_3)) * (float)device.gyro.dt;
		SEq_4 += (SEqDot_omega_4 - (beta * SEqHatDot_4)) * (float)device.gyro.dt;

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

		//std::cout << SEq_1 << " " << SEq_2 << " " << SEq_3 << " " << SEq_4 << std::endl;
		//std::cout << device.gyro.x << " " << device.gyro.y << " " << device.gyro.z << std::endl;
	}
	
	return 0;
}

int Rslam::solveCameraPose() {
	// Get color and depth frames
	extractDepth(device0);
	extractIr(device0);
	upsampleDepth(device0);

	//extractDepth(device1);
	//extractIr(device1);
	//upsampleDepth(device1);

	//visualizeColor(device0);
	//visualizeDepth(device0);

	// Solve current frame keypoints nad descriptors
	detectAndComputeOrb(device0.infrared1, device0.d_ir1, device0.keypointsIr1, device0.d_descriptorsIr1);
	//detectAndComputeOrb(device1.infrared1, device1.d_ir1, device1.keypointsIr1, device1.d_descriptorsIr1);

	// Match with keyframe
	matchAndPose(device0);
	//matchAndPose(device1);
	visualizeRelativeKeypoints(device0.currentKeyframe, device0.infrared1, "dev0");
	//visualizeRelativeKeypoints(device1.currentKeyframe, device1.infrared1, "dev1");

	Device viewDevice = device0;
	viewDevice.Rvec = viewDevice.currentKeyframe->R;
	viewDevice.t = viewDevice.currentKeyframe->t;
	this->Rvec = viewDevice.Rvec;
	this->t = viewDevice.t;
	cv::Mat im = cv::Mat::zeros(100, 300, CV_8UC3);
	im.setTo(cv::Scalar(50, 50, 50));
	overlayMatrix("pose", im, this->Rvec, this->t);

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
					keyframe->matchedKeypoints.push_back(keyframe->keypoints[device.matches[k][0].queryIdx]);
					keyframe->matchedKeypointsSrc.push_back(device.keypointsIr1[device.matches[k][0].trainIdx]);

					keyframe->matchedPoints.push_back(keyframe->keypoints[device.matches[k][0].queryIdx].pt);
					cv::Point2f srcPt = device.keypointsIr1[device.matches[k][0].trainIdx].pt;
					//keyframe->matchedPointsSrc.push_back(keypointsIr1[matches[k][0].trainIdx].pt);
					keyframe->matchedPointsSrc.push_back(srcPt);

					keyframe->matchedDistances.push_back(device.matches[k][0].distance);

					// Get corresponding 3D point
					double z = ((double)device.depth.at<short>(srcPt))/256.0;
					// Solve 3D point
					cv::Point3f src3dpt;
					src3dpt.x = (float)(((double)srcPt.x - device.cx) * z / device.fx);
					src3dpt.y = (float)(((double)srcPt.y - device.cy) * z / device.fy);
					src3dpt.z = (float) z;
					keyframe->objectPointsSrc.push_back(src3dpt);
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
		cv::solvePnPRansac(keyframe->objectPointsSrc, keyframe->matchedPoints, device.intrinsic, device.distCoeffs, keyframe->R, keyframe->t);
		//cv::solvePnP(keyframe->objectPointsSrc, keyframe->matchedPoints, this->intrinsic, this->distCoeffs, keyframe->R, keyframe->t, false);
	}
	// Convert R and t to the current frame
	return 0;
}

int Rslam::createImuKeyframe(Device &device) {
	if (!device.imuKeyframeExist) {
		device.ImuRotation.x = 0.0f;
		device.ImuRotation.y = 0.0f;
		device.ImuRotation.z = 0.0f;
		device.ImuRotation.w = 1.0f;
	}
	return 0;
}

// Fetch Frames
int Rslam::extractGyroAndAccel(Device &device) {
	device.gyro.lastTs = device.gyro.ts;
	auto gyroFrame = device.frameset.first(RS2_STREAM_GYRO, RS2_FORMAT_MOTION_XYZ32F).as<rs2::motion_frame>();
	rs2_vector gv = gyroFrame.get_motion_data();
	device.gyro.x = gv.x - GYRO_BIAS_X;
	device.gyro.y = gv.y - GYRO_BIAS_Y;
	device.gyro.z = gv.z - GYRO_BIAS_Z;
	device.gyro.ts = gyroFrame.get_timestamp();
	device.gyro.dt = (device.gyro.ts - device.gyro.lastTs) / 1000.0;
	//std::cout << std::fixed
	//			<< device.gyro.ts << " " << device.gyro.lastTs << " "
	//			<< device.gyro.dt << ": ("
	//			/*<< device.gyro.x << ","
	//			<< device.gyro.y << ","
	//			<< device.gyro.z << " )"
	//			<< device.accel.dt << ": ("
	//			<< device.accel.x << " "
	//			<< device.accel.y << " "
	//			<< device.accel.z << ")"*/
	//			<< std::endl;

	device.accel.lastTs = device.accel.ts;
	auto accelFrame = device.frameset.first(RS2_STREAM_ACCEL, RS2_FORMAT_MOTION_XYZ32F).as<rs2::motion_frame>();
	rs2_vector av = accelFrame.get_motion_data();
	device.accel.x = av.x;
	device.accel.y = av.y;
	device.accel.z = av.z;
	device.accel.ts = accelFrame.get_timestamp();
	device.accel.dt = (device.accel.ts - device.accel.lastTs) / 1000.0;

	//std::cout << (float)device.accel.dt << std::endl;
	//float R = sqrtf(av.x * av.x + av.y * av.y + av.z * av.z);
	//float newRoll = acos(av.x / R);
	//float newYaw = acos(av.y / R);
	//float newPitch = acos(av.z / R);
	//std::cout << accel.dt << std::endl;
	return 0;
}

int Rslam::extractColor(Device &device) {
	auto colorData = device.frameset.get_color_frame();
	device.color = cv::Mat(cv::Size(width, height), CV_8UC3, (void*)colorData.get_data(), cv::Mat::AUTO_STEP);
	return 0;
}

int Rslam::extractDepth(Device &device) {
	auto depthData = device.frameset.get_depth_frame();
	//rs2::frame filtered = depthData;
	//filtered = spatialFilter.process(filtered);
	device.depth = cv::Mat(cv::Size(width, height), CV_16S, (void*)depthData.get_data(), cv::Mat::AUTO_STEP);
	//device.depth = cv::Mat(cv::Size(width, height), CV_16S, (void*)filtered.get_data(), cv::Mat::AUTO_STEP);
	return 0;
}

int Rslam::extractIr(Device &device) {
	auto infrared1Data = device.frameset.get_infrared_frame(1);
	device.infrared1 = cv::Mat(cv::Size(width, height), CV_8UC1, (void*)infrared1Data.get_data(), cv::Mat::AUTO_STEP);
	device.infrared1.convertTo(device.infrared132f, CV_32F, 1 / 256.0f);

	auto infrared2Data = device.frameset.get_infrared_frame(2);
	device.infrared2 = cv::Mat(cv::Size(width, height), CV_8UC1, (void*)infrared2Data.get_data(), cv::Mat::AUTO_STEP);
	device.infrared2.convertTo(device.infrared232f, CV_32F, 1 / 256.0f);
	return 0;
}

int Rslam::upsampleDepth(Device &device) {
	//std::cout << device.depth.at<short int>(320, 160) << std::endl;
	//cv::imshow("test1", device.depth);
	device.depth.convertTo(device.depth32f, CV_32F, 1.0f / 256.0f);
	upsampling->copyImagesToDevice(device.infrared132f, device.depth32f);
	upsampling->propagateColorOnly();
	//upsampling->solve();
	upsampling->copyImagesToHost(device.depth32f);
	//cv::imshow("test2", device.depth32f);
	//cv::imshow("test", device.depth32f);
	device.depth32f.convertTo(device.depth, CV_16S, 256.0f);
	//cv::imshow("test2", device.depth);
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
	float scale = 0.003f;
	objFile->readObj("arrow.obj", FileReader::ArrayFormat::VERTEX_NORMAL_TEXTURE, scale);

	//Axis for Camera pose
	//cv::Mat texture = cv::imread("texture.png");
	CgObject *camAxis = new CgObject();
	camAxis->objectIndex = 0;
	camAxis->loadShader("myshader2.vert", "myshader2.frag");
	camAxis->loadData(objFile->vertexArray, objFile->indexArray, CgObject::ArrayFormat::VERTEX_NORMAL_TEXTURE);
	camAxis->loadTexture("axiscolor.png");
	camAxis->setDrawMode(CgObject::Mode::TRIANGLES);
	camAxis->setLight();
	viewer->cgObject->push_back(camAxis);

	//Axis for IMU pose
	//cv::Mat texture = cv::imread("texture.png");
	CgObject *imuAxis = new CgObject();
	imuAxis->objectIndex = 0;
	imuAxis->loadShader("myshader2.vert", "myshader2.frag");
	imuAxis->loadData(objFile->vertexArray, objFile->indexArray, CgObject::ArrayFormat::VERTEX_NORMAL_TEXTURE);
	imuAxis->loadTexture("imuaxiscolor.png");
	imuAxis->setDrawMode(CgObject::Mode::TRIANGLES);
	imuAxis->setLight();
	viewer->cgObject->push_back(imuAxis);

	viewer->run();
	viewer->close();
}

void Rslam::toEuler(Quaternion q, Vector3 &euler)
{
	// roll (x-axis rotation)
	float sinr_cosp = +2.0f * (q.w * q.x + q.y * q.z);
	float cosr_cosp = +1.0f - 2.0f * (q.x * q.x + q.y * q.y);
	euler.x = atan2(sinr_cosp, cosr_cosp);

	// pitch (y-axis rotation)
	float sinp = 2.0f * (q.w * q.y - q.z * q.x);
	if (fabs(sinp) >= 1.0f)
		euler.y = copysign(3.14159f / 2.0f, sinp); // use 90 degrees if out of range
	else
		euler.y = asin(sinp);

	// yaw (z-axis rotation)
	float siny_cosp = 2.0f * (q.w * q.z + q.x * q.y);
	float cosy_cosp = 1.0f - 2.0f * (q.y * q.y + q.z * q.z);
	euler.z = atan2(siny_cosp, cosy_cosp);
}

void Rslam::toQuaternion(Vector3 euler, Quaternion &q) // yaw (Z), pitch (Y), roll (X)
{
	// Abbreviations for the various angular functions
	double cy = cos(euler.z * 0.5);
	double sy = sin(euler.z * 0.5);
	double cp = cos(euler.y * 0.5);
	double sp = sin(euler.y * 0.5);
	double cr = cos(euler.x * 0.5);
	double sr = sin(euler.x * 0.5);

	q.w = cy * cp * cr + sy * sp * sr;
	q.x = cy * cp * sr - sy * sp * cr;
	q.y = sy * cp * sr + cy * sp * cr;
	q.z = sy * cp * cr - cy * sp * sr;
}

void Rslam::updateViewerPose(Device &device) {
	if (viewer->isRunning) {
		//update camaxis
		viewer->cgObject->at(0)->rx = -(float)Rvec.at<double>(1);
		viewer->cgObject->at(0)->ry = (float)Rvec.at<double>(0);
		viewer->cgObject->at(0)->rz = -(float)Rvec.at<double>(2);
		viewer->cgObject->at(0)->tx = (float)t.at<double>(0) * 10.0f;
		viewer->cgObject->at(0)->ty = -(float)t.at<double>(1) * 10.0f;
		viewer->cgObject->at(0)->tz = -(float)t.at<double>(2) *10.0f;

		// update imuaxis
		Vector3 euler;
		toEuler(device.ImuRotation, euler);
		viewer->cgObject->at(1)->rx = euler.z;
		viewer->cgObject->at(1)->ry = euler.x + (3.14159265358979f / 2.0f);
		viewer->cgObject->at(1)->rz = -euler.y;// +(3.14159265358979f / 2.0f);
		viewer->cgObject->at(1)->tx = (float)t.at<double>(0) * 10.0f;
		viewer->cgObject->at(1)->ty = -(float)t.at<double>(1) * 10.0f;
		viewer->cgObject->at(1)->tz = -(float)t.at<double>(2) *10.0f;

		cv::Mat im = cv::Mat::zeros(100, 300, CV_8UC3);
		im.setTo(cv::Scalar(50, 50, 50));
		double imuRvecdata[3] = { (double)euler.x, (double)euler.y, (double)euler.z };
		cv::Mat imuRvec = cv::Mat(1, 3, CV_64F, imuRvecdata).clone();
		overlayMatrix("imupose", im, imuRvec, imuRvec);
	}
}

void Rslam::updateViewerPose() {
	// Global
	if (viewer->isRunning) {
		//update camaxis
		viewer->cgObject->at(0)->rx = -(float)Rvec.at<double>(1);
		viewer->cgObject->at(0)->ry = (float)Rvec.at<double>(0);
		viewer->cgObject->at(0)->rz = -(float)Rvec.at<double>(2);
		viewer->cgObject->at(0)->tx = (float)t.at<double>(0) * 10.0f;
		viewer->cgObject->at(0)->ty = -(float)t.at<double>(1) * 10.0f;
		viewer->cgObject->at(0)->tz = -(float)t.at<double>(2) *10.0f;
	}
}

void Rslam::visualizeRelativeKeypoints(Keyframe *keyframe, cv::Mat ir1) {
	visualizeRelativeKeypoints(keyframe, ir1, "default");
}

void Rslam::visualizeRelativeKeypoints(Keyframe *keyframe, cv::Mat ir1, std::string windowNamePrefix) {
	cv::Mat imout1, imout2;
	cv::drawKeypoints(keyframe->im, keyframe->matchedKeypoints, imout1, cv::Scalar(0, 255, 0), cv::DrawMatchesFlags::DEFAULT);
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
	device.depth32f.convertTo(depth8u, CV_8UC1, 256.0f / this->maxDepth);
	cv::applyColorMap(depth8u, cm_img0, cv::COLORMAP_JET);
	// Show the result:
	cv::imshow("cm_img0", cm_img0);
}

void Rslam::overlayMatrix(const char* windowName, cv::Mat &im, cv::Mat R1, cv::Mat t) {
	std::ostringstream message1, message2, message3;
	int precision = 3;
	message1 << std::fixed << this->parseDecimal(R1.at<double>(0), precision) << " " << this->parseDecimal(R1.at<double>(1), precision) << " " << this->parseDecimal(R1.at<double>(2), precision);// << " " << this->parseDecimal(t.at<double>(0));
	message2 << std::fixed << this->parseDecimal(t.at<double>(0), precision) << " " << this->parseDecimal(t.at<double>(1), precision) << " " << this->parseDecimal(t.at<double>(2), precision);// << " " << this->parseDecimal(t.at<double>(1));
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









































// Unused
int Rslam::poseSolver() {
	//double last_ts[RS2_STREAM_COUNT];
	//double dt[RS2_STREAM_COUNT];
	//std::clock_t start;

	//double timer = 0.0;
	//double fps, oldFps = 0.0;

	//cv::Mat prevInfrared1 = cv::Mat::zeros(cv::Size(this->width, this->height), CV_8UC1);
	//Keyframe * currentKeyframe = new Keyframe();
	//bool keyframeExist = false;

	//while (true) {
	//	char pressed = cv::waitKey(10);
	//	if (pressed == 27) break;
	//	if (pressed == 'r') keyframeExist = false; // change this to automatic keyframing

	//	bool pollSuccess = pipelines[0]->poll_for_frames(&frameset);
	//	if (!pollSuccess) continue;

	//	start = std::clock();
	//	frameset = alignToColor.process(frameset);
	//	//extractTimeStamps();
	//	extractGyroAndAccel();
	//	visualizeImu();
	//	updatePose();

	//	// Get color and depth frames
	//	//extractColorAndDepth();
	//	extractIr();

	//	//visualizeColor();
	//	//visualizeDepth();

	//	// Solve Stereo
	//	if (featMethod == SURF) {
	//		solveStereoSurf(infrared1, infrared2);
	//	}
	//	else if (featMethod == ORB) {
	//		solveStereoOrb(infrared1, infrared2);
	//	}
	//	//visualizeStereoKeypoints(infrared1, infrared2);

	//	// Solve Pose
	//	// Add keypoints from stereo that are not in the current keyframe
	//	if (!keyframeExist) {
	//		currentKeyframe->im = infrared1.clone();
	//		currentKeyframe->d_im.upload(currentKeyframe->im);
	//		currentKeyframe->keypoints = keypointsIr1;
	//		//std::cout << "Keyframe keypoints: " << currentKeyframe->keypoints.size() << std::endl;
	//		keyframeExist = true;
	//	}
	//	else {
	//		if (featMethod == SURF) {
	//			solveRelativeSurf(currentKeyframe);
	//		}
	//		else {
	//			solveRelativeOrb(currentKeyframe);
	//		}
	//		visualizeRelativeKeypoints(currentKeyframe, infrared1);
	//		//currentKeyframe->im = infrared1.clone();
	//		//currentKeyframe->keypoints = stereoKeypointsIr1;
	//		//currentKeyframe->keypoints = currentKeyframe->matchedKeypoints;
	//		//std::cout << currentKeyframe->keypoints.size() << std::endl;
	//	}

	//	fps = 1 / ((std::clock() - start) / (double)CLOCKS_PER_SEC);
	//	oldFps = (fps + oldFps) / 2.0; // Running average
	//	visualizeFps(oldFps);
	//	/*if (oldFps > 60.0) {
	//		visualizeFps(oldFps);
	//	}
	//	else {
	//		visualizeFps(oldFps);
	//	}*/

	//	/*std::cout << std::fixed
	//		<< gyro.ts << " " << gyro.lastTs << " "
	//		<< gyro.dt << ": ("
	//		<< gyro.x << ","
	//		<< gyro.y << ","
	//		<< gyro.z << " )"
	//		<< accel.dt << ": ("
	//		<< accel.x << " "
	//		<< accel.y << " "
	//		<< accel.z << ")"
	//		<< std::endl;*/
	//}
	//// poll for frames (gyro, accel)
	//// if depth and image available, fetch depth and image
	//// set first stable frame as reference
	//// align depth with image
	//// feature extraction
	//// feature matching
	//// pose estimation
	//return 0;
}

int Rslam::poseSolverDefaultStereo() {
	//try {
	//	double last_ts[RS2_STREAM_COUNT];
	//	double dt[RS2_STREAM_COUNT];
	//	std::clock_t start;

	//	double timer = 0.0;
	//	double fps, oldFps = 0.0;

	//	cv::Mat prevInfrared1 = cv::Mat::zeros(cv::Size(this->width, this->height), CV_8UC1);
	//	Keyframe * currentKeyframe = new Keyframe();
	//	bool keyframeExist = false;
	//	currentKeyframe->R = cv::Mat::zeros(3, 1, CV_64F);
	//	currentKeyframe->t = cv::Mat::zeros(3, 1, CV_64F);
	//	std::cout << "Pose Solver thread started: " << pipelines.size() << std::endl;
	//	while (true) {
	//		char pressed = cv::waitKey(10);
	//		if (pressed == 27) break;
	//		if (pressed == 'r') keyframeExist = false; // change this to automatic keyframing

	//		//bool pollSuccess = pipelines[0]->poll_for_frames(&frameset);
	//		bool pollSuccess = device0.pipe->poll_for_frames(&device0.frameset);
	//		//std::cout << pollSuccess;
	//		if (!pollSuccess) continue;

	//		start = std::clock();
	//		device0.frameset = alignToColor.process(device0.frameset);
	//		extractGyroAndAccel(device0);

	//		//visualizeImu();

	//		// Get color and depth frames
	//		extractColorAndDepth(device0);
	//		extractIr(device0);
	//		//visualizeColor();
	//		//visualizeDepth();

	//		// Solve current frame keypoints nad descriptors
	//		detectAndComputeOrb(device0.infrared1, device0.d_ir1, device0.keypointsIr1, device0.d_descriptorsIr1);

	//		// Match with keyframe
	//		if (!keyframeExist) {
	//			currentKeyframe->im = device0.infrared1.clone();
	//			currentKeyframe->d_im.upload(currentKeyframe->im);
	//			currentKeyframe->keypoints = device0.keypointsIr1;
	//			currentKeyframe->d_descriptors = device0.d_descriptorsIr1.clone();
	//			keyframeExist = true;
	//		}
	//		else {
	//			/*if (featMethod == SURF) {
	//				solveRelativeSurf(currentKeyframe);
	//			}
	//			else {
	//				relativeMatchingDefaultStereo(currentKeyframe, infrared1);
	//			}*/
	//			relativeMatchingDefaultStereo(device0, currentKeyframe, device0.infrared1);
	//			//visualizeRelativeKeypoints(currentKeyframe, infrared1);
	//			solveRelativePose(device0, currentKeyframe);

	//			device0.Rvec = currentKeyframe->R;
	//			device0.t = currentKeyframe->t;
	//			updateViewerPose();
	//		}

	//		fps = 1 / ((std::clock() - start) / (double)CLOCKS_PER_SEC);
	//		oldFps = (fps + oldFps) / 2.0; // Running average
	//		visualizeFps(oldFps);

	//	}
	//}
	//catch (const rs2::error & e)
	//{
	//	std::cerr << "RealSense error calling " << e.get_failed_function() << "(" << e.get_failed_args() << "):\n    " << e.what() << std::endl;
	//	return EXIT_FAILURE;
	//}
	//catch (const std::exception& e)
	//{
	//	std::cerr << e.what() << std::endl;
	//	return EXIT_FAILURE;
	//}
	return EXIT_SUCCESS;
}

void Rslam::visualizeStereoKeypoints(cv::Mat ir1, cv::Mat ir2) {
	/*cv::Mat imout1, imout2;
	cv::drawKeypoints(ir1, stereoKeypointsIr1, imout1, cv::Scalar(0, 255, 0), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	cv::putText(imout1, "detected keypoints: " + parseDecimal((double)keypointsIr1.size(), 0), cv::Point(0, 10), cv::FONT_HERSHEY_PLAIN, 1, CV_RGB(255, 255, 255));
	cv::putText(imout1, "matched keypoints: " + parseDecimal((double)stereoKeypointsIr1.size(), 0), cv::Point(0, 30), cv::FONT_HERSHEY_PLAIN, 1, CV_RGB(255, 255, 255));
	cv::imshow("ir1", imout1);
	cv::drawKeypoints(ir2, stereoKeypointsIr2, imout2, cv::Scalar(0, 255, 0), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	cv::imshow("ir2", imout2);*/
}



void Rslam::visualizeKeypoints(cv::Mat im) {
	/*cv::Mat imout;
	cv::drawKeypoints(im, keypoints, imout, cv::Scalar(0, 255, 0), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	cv::imshow("keypoints", imout);*/
}

void Rslam::visualizeKeypoints(cv::Mat ir1, cv::Mat ir2) {
	/*cv::Mat imout1, imout2;
	cv::drawKeypoints(ir1, keypointsIr1, imout1, cv::Scalar(0, 255, 0), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	cv::imshow("ir1", imout1);
	cv::drawKeypoints(ir2, keypointsIr2, imout2, cv::Scalar(0, 255, 0), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	cv::imshow("ir2", imout1);*/
}

void Rslam::updatePose() {
	/*pose.x = accel.x;
	pose.y = accel.y;
	pose.z = accel.z;
	pose.rx = 0.0f;
	pose.ry = 0.0f;
	pose.rz = 0.0f;
	pose.rw = 1.0f;*/
}

int Rslam::solveStereoSurf(cv::Mat ir1, cv::Mat ir2) {
	//d_ir1.upload(ir1); //first cuda call is always slow
	//d_ir2.upload(ir2);
	//surf(d_ir1, cv::cuda::GpuMat(), d_keypointsIr1, d_descriptorsIr1);
	//surf(d_ir2, cv::cuda::GpuMat(), d_keypointsIr2, d_descriptorsIr2);
	//surf.downloadKeypoints(d_keypointsIr1, keypointsIr1);
	//surf.downloadKeypoints(d_keypointsIr2, keypointsIr2);

	//if ((d_keypointsIr1.empty() || d_keypointsIr2.empty()) || (d_descriptorsIr1.cols <= 1) || (d_descriptorsIr2.cols <= 1)) {
	//	std::cout << "No keypoints found." << std::endl;
	//}
	//else {
	//	matcher->knnMatch(d_descriptorsIr1, d_descriptorsIr2, matches, 2);
	//	if (!matches.empty()) {
	//		stereoKeypointsIr1 = std::vector< cv::KeyPoint >();
	//		stereoKeypointsIr2 = std::vector< cv::KeyPoint >();
	//		stereoKeypointsIr1.clear();
	//		stereoKeypointsIr2.clear();
	//		stereoPointsIr1 = std::vector<cv::Point2f>();
	//		stereoPointsIr2 = std::vector<cv::Point2f>();
	//		stereoPointsIr1.clear();
	//		stereoPointsIr2.clear();
	//		stereoDistances = std::vector< float >();
	//		stereoDistances.clear();
	//		for (int k = 0; k < (int)matches.size(); k++)
	//		{
	//			if ((matches[k][0].distance < 0.6*(matches[k][1].distance)) && ((int)matches[k].size() <= 2 && (int)matches[k].size() > 0))
	//			{
	//				stereoKeypointsIr1.push_back(keypointsIr1[matches[k][0].queryIdx]);
	//				stereoKeypointsIr2.push_back(keypointsIr2[matches[k][0].trainIdx]);
	//				stereoPointsIr1.push_back(keypointsIr1[matches[k][0].queryIdx].pt);
	//				stereoPointsIr2.push_back(keypointsIr2[matches[k][0].trainIdx].pt);
	//				stereoDistances.push_back(matches[k][0].distance);
	//			}
	//		}
	//	}
	//}
	return 0;
}

int Rslam::solveStereoOrb(cv::Mat ir1, cv::Mat ir2) {
	//d_ir1.upload(ir1); //first cuda call is always slow
	//d_ir2.upload(ir2);

	///*orb->detect(d_ir1, keypointsIr1);
	//orb->compute(d_ir1, keypointsIr1, d_descriptorsIr1);

	//orb->detect(d_ir2, keypointsIr2);
	//orb->compute(d_ir2, keypointsIr2, d_descriptorsIr2);*/
	//orb->detectAndCompute(d_ir1, cv::cuda::GpuMat(), keypointsIr1, d_descriptorsIr1);
	//orb->detectAndCompute(d_ir2, cv::cuda::GpuMat(), keypointsIr2, d_descriptorsIr2);
	////d_descriptorsIr1.download(descriptorsIr1);
	////d_descriptorsIr2.download(descriptorsIr2);

	//if ((keypointsIr1.empty() || keypointsIr2.empty()) || (d_descriptorsIr1.cols <= 1) || (d_descriptorsIr2.cols <= 1)) {
	//	std::cout << "No keypoints found." << std::endl;
	//}
	//else {
	//	matcher->knnMatch(d_descriptorsIr1, d_descriptorsIr2, matches, 2);
	//	if (!matches.empty()) {
	//		stereoKeypointsIr1 = std::vector< cv::KeyPoint >();
	//		stereoKeypointsIr2 = std::vector< cv::KeyPoint >();
	//		stereoKeypointsIr1.clear();
	//		stereoKeypointsIr2.clear();
	//		stereoPointsIr1 = std::vector<cv::Point2f>();
	//		stereoPointsIr2 = std::vector<cv::Point2f>();
	//		stereoPointsIr1.clear();
	//		stereoPointsIr2.clear();
	//		stereoDistances = std::vector< float >();
	//		stereoDistances.clear();
	//		for (int k = 0; k < (int)matches.size(); k++)
	//		{
	//			if ((matches[k][0].distance < 0.6*(matches[k][1].distance)) && ((int)matches[k].size() <= 2 && (int)matches[k].size() > 0))
	//			{
	//				stereoKeypointsIr1.push_back(keypointsIr1[matches[k][0].queryIdx]);
	//				stereoKeypointsIr2.push_back(keypointsIr2[matches[k][0].trainIdx]);
	//				stereoPointsIr1.push_back(keypointsIr1[matches[k][0].queryIdx].pt);
	//				stereoPointsIr2.push_back(keypointsIr2[matches[k][0].trainIdx].pt);
	//				stereoDistances.push_back(matches[k][0].distance);
	//			}
	//		}
	//	}
	//}
	return 0;
}

int Rslam::solveRelativeOrb(Keyframe *keyframe) {
	//// Recompute descriptors
	///*if ((keyframe->keypoints.empty() || stereoKeypointsIr1.empty())) return 0;

	//orb->compute(keyframe->d_im, keyframe->keypoints, keyframe->d_descriptors);
	//orb->compute(d_ir1, stereoKeypointsIr1, d_descriptorsIr1);*/
	////orb->detect(keyframe->d_im, keyframe->keypoints);
	//if (!keyframe->keypoints.empty()) {

	//	orb->compute(keyframe->d_im, keyframe->keypoints, keyframe->d_descriptors);
	//	//std::cout << keyframe->keypoints.size() << std::endl;
	//}

	////keyframe->keypoints.clear();
	////orb->detectAndCompute(keyframe->d_im, cv::cuda::GpuMat(), keyframe->keypoints, keyframe->d_descriptors);
	//if (!stereoKeypointsIr1.empty()) {
	//	orb->compute(d_ir1, stereoKeypointsIr1, d_descriptorsIr1);
	//}

	//if ((stereoKeypointsIr1.empty() || keyframe->keypoints.empty()) || (d_descriptorsIr1.cols <= 1) || (keyframe->d_descriptors.cols <= 1)) {
	//	std::cout << "No keypoints found for relative pose." << std::endl;
	//}
	//else {
	//	matcher->knnMatch(keyframe->d_descriptors, d_descriptorsIr1, matches, 2);
	//	if (!matches.empty()) {
	//		//std::cout << "Matches: " << matches.size() << std::endl;
	//		keyframe->matchedKeypoints = std::vector< cv::KeyPoint >();
	//		keyframe->matchedKeypointsSrc = std::vector< cv::KeyPoint >();
	//		keyframe->matchedKeypoints.clear();
	//		keyframe->matchedKeypointsSrc.clear();

	//		keyframe->matchedPoints = std::vector<cv::Point2f>();
	//		keyframe->matchedPointsSrc = std::vector<cv::Point2f>();
	//		keyframe->matchedPoints.clear();
	//		keyframe->matchedPointsSrc.clear();

	//		keyframe->matchedDistances = std::vector< float >();
	//		keyframe->matchedDistances.clear();

	//		for (int k = 0; k < (int)matches.size(); k++)
	//		{
	//			if ((matches[k][0].distance < 0.6*(matches[k][1].distance)) && ((int)matches[k].size() <= 2 && (int)matches[k].size() > 0))
	//			{
	//				keyframe->matchedKeypoints.push_back(keyframe->keypoints[matches[k][0].queryIdx]);
	//				keyframe->matchedKeypointsSrc.push_back(stereoKeypointsIr1[matches[k][0].trainIdx]);

	//				keyframe->matchedPoints.push_back(keyframe->keypoints[matches[k][0].queryIdx].pt);
	//				keyframe->matchedPointsSrc.push_back(stereoKeypointsIr1[matches[k][0].trainIdx].pt);

	//				keyframe->matchedDistances.push_back(matches[k][0].distance);
	//			}
	//		}
	//	}
	//	else {
	//		std::cout << "No relative matches found. " << std::endl;
	//	}

	//}
	return 0;
}

int Rslam::detectAndComputeSurf(cv::Mat im, cv::cuda::GpuMat &d_im, std::vector<cv::KeyPoint> &keypoints, cv::cuda::GpuMat &descriptors) {
	return 0;
}

int Rslam::solveRelativeSurf(Keyframe *keyframe) {
	//keyframe->d_im.upload(keyframe->im);

	//// Recompute descriptors
	//surf(keyframe->d_im, cv::cuda::GpuMat(), keyframe->keypoints, keyframe->d_descriptors, true);
	//surf(d_ir1, cv::cuda::GpuMat(), stereoKeypointsIr1, d_descriptorsIr1, true);
	//surf.downloadKeypoints(d_keypointsIr1, keypointsIr1);
	//surf.downloadKeypoints(d_keypointsIr2, keypointsIr2);

	//if ((stereoKeypointsIr1.empty() || keyframe->keypoints.empty()) || (d_descriptorsIr1.cols <= 1) || (keyframe->d_descriptors.cols <= 1)) {
	//	std::cout << "No keypoints found for relative pose." << std::endl;
	//}
	//else {
	//	matcher->knnMatch(d_descriptorsIr1, keyframe->d_descriptors, matches, 2);
	//	if (!matches.empty()) {
	//		//std::cout << "Matches: " << matches.size() << std::endl;
	//		keyframe->matchedKeypoints = std::vector< cv::KeyPoint >();
	//		keyframe->matchedKeypointsSrc = std::vector< cv::KeyPoint >();
	//		keyframe->matchedKeypoints.clear();
	//		keyframe->matchedKeypointsSrc.clear();
	//		keyframe->matchedPoints = std::vector<cv::Point2f>();
	//		keyframe->matchedPointsSrc = std::vector<cv::Point2f>();
	//		keyframe->matchedPoints.clear();
	//		keyframe->matchedPointsSrc.clear();
	//		keyframe->matchedDistances = std::vector< float >();
	//		keyframe->matchedDistances.clear();
	//		for (int k = 0; k < (int)matches.size(); k++)
	//		{
	//			if ((matches[k][0].distance < 0.6*(matches[k][1].distance)) && ((int)matches[k].size() <= 2 && (int)matches[k].size() > 0))
	//			{
	//				keyframe->matchedKeypoints.push_back(keyframe->keypoints[matches[k][0].queryIdx]);
	//				keyframe->matchedKeypointsSrc.push_back(stereoKeypointsIr1[matches[k][0].trainIdx]);
	//				keyframe->matchedPoints.push_back(keyframe->keypoints[matches[k][0].queryIdx].pt);
	//				keyframe->matchedPointsSrc.push_back(stereoKeypointsIr1[matches[k][0].trainIdx].pt);
	//				keyframe->matchedDistances.push_back(matches[k][0].distance);
	//			}
	//		}
	//	}
	//	else {
	//		std::cout << "No relative matches found. " << std::endl;
	//	}

	//}
	return 0;
}

int Rslam::solveKeypointsAndDescriptors(cv::Mat im) {
	////cv::Mat gray;
	////cv::cvtColor(im, gray, CV_BGR2GRAY);
	////d_im.upload(gray); //first cuda call is always slow
	//d_im.upload(im); //first cuda call is always slow
	//surf(d_im, cv::cuda::GpuMat(), d_keypoints, d_descriptors);
	//surf.downloadKeypoints(d_keypoints, keypoints);
	//return 0;
}




















// TESTS **********************************************
int Rslam::testOrb() {
	cv::Mat im = cv::imread("ir1.png");
	cv::Ptr<cv::cuda::ORB> orb = cv::cuda::ORB::create();
	cv::Mat gray;
	cv::cvtColor(im, gray, CV_BGR2GRAY);
	cv::cuda::GpuMat d_im(gray);
	std::vector<cv::KeyPoint> kp;
	orb->detect(d_im, kp);
	cv::Mat imout;
	cv::drawKeypoints(im, kp, imout, cv::Scalar(0, 255, 0), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	cv::imshow("orb test", imout);
	cv::waitKey();
	return 0;
}

int Rslam::testT265() {
	try {
		rs2::context ctx;
		rs2::pipeline pipe(ctx);
		rs2::config cfg;
		auto dev = ctx.query_devices();
		cfg.enable_device("852212110449");
		pipe.start(cfg);

		cv::namedWindow("fisheye1", cv::WINDOW_AUTOSIZE);
		cv::namedWindow("fisheye2", cv::WINDOW_AUTOSIZE);

		while (cv::waitKey(1) < 0 && cv::getWindowProperty("fisheye1", cv::WND_PROP_AUTOSIZE) >= 0)
		{
			rs2::frameset frameset = pipe.wait_for_frames();
			auto fisheye1 = frameset.get_fisheye_frame(1);
			auto fisheye2 = frameset.get_fisheye_frame(2);
			const int w = fisheye1.as<rs2::video_frame>().get_width();
			const int h = fisheye1.as<rs2::video_frame>().get_height();
			cv::Mat fs1(cv::Size(w, h), CV_8UC1, (void*)fisheye1.get_data(), cv::Mat::AUTO_STEP);
			cv::Mat fs2(cv::Size(w, h), CV_8UC1, (void*)fisheye2.get_data(), cv::Mat::AUTO_STEP);
			cv::imshow("fisheye1", fs1);
			cv::imshow("fisheye2", fs2);
		}
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
}

int Rslam::runTestViewerSimpleThread() {
	std::thread t1(&Rslam::testViewerSimple, this);
	std::thread t2(&Rslam::poseSolver, this);
	t1.join();
	t2.join();
}

int Rslam::testViewerSimple() {
	Viewer *viewer = new Viewer();
	viewer->createWindow(800, 600, "test");
	viewer->setCameraProjectionType(Viewer::ProjectionType::PERSPECTIVE);

	FileReader *objFile = new FileReader();
	objFile->readObj("D:/dev/c-projects/OpenSor/test_opensor_viewer/data/models/monkey.obj", FileReader::ArrayFormat::VERTEX_NORMAL_TEXTURE, 0.01f);
	//std::cout << objFile->vertexArray.size() << std::endl;
	//std::cout << objFile->indexArray.size() << std::endl;

	//box solid
	//cv::Mat texture = cv::imread("texture.png");
	CgObject *cgobject = new CgObject();
	cgobject->loadShader("myshader2.vert", "myshader2.frag");
	cgobject->loadData(objFile->vertexArray, objFile->indexArray, CgObject::ArrayFormat::VERTEX_NORMAL_TEXTURE);
	cgobject->loadTexture("default_texture.jpg");
	cgobject->setDrawMode(CgObject::Mode::TRIANGLES);
	viewer->cgObject->push_back(cgobject);

	viewer->run();
	//viewer->depthAndVertexCapture();
	viewer->close();

	return 0;
}


int Rslam::getFrames() {
	/*rs2::pipeline pipe;
	frameset = pipe.wait_for_frames();
	frameset = alignToColor.process(frameset);
	auto depthData = frameset.get_depth_frame();
	auto colorData = frameset.get_color_frame();
	auto depthVisData = colorizer.colorize(depthData);
	depthVis = cv::Mat(cv::Size(width, height), CV_8UC3, (void*)depthVisData.get_data(), cv::Mat::AUTO_STEP);
	color = cv::Mat(cv::Size(width, height), CV_8UC3, (void*)colorData.get_data(), cv::Mat::AUTO_STEP);*/
	return 0;
}

int Rslam::getPose(float *x, float *y, float *z, float *roll, float *pitch, float *yaw) {
	return 0;
}

int Rslam::getGyro(float *rateRoll, float *ratePitch, float *rateYaw) {
	//rs2::pipeline pipe;
	//frameset = pipe.wait_for_frames();
	//rs2_vector gv = frameset.first(RS2_STREAM_GYRO, RS2_FORMAT_MOTION_XYZ32F).as<rs2::motion_frame>().get_motion_data();
	//gyro.x = gv.x - GYRO_BIAS_X;
	//gyro.y = gv.y - GYRO_BIAS_Y;
	//gyro.z = gv.z - GYRO_BIAS_Z;
	//*rateRoll = gyro.x;
	//*ratePitch = gyro.y;
	//*rateYaw = gyro.z;
	////std::cout << gyroRatePitch << std::endl;
	return 0;
}

int Rslam::testStream() {
	/*while (cv::waitKey(1) < 0)
	{
		getFrames();
		cv::imshow("depth", depthVis);
		cv::imshow("color", color);
		cv::Mat combined;
		cv::addWeighted(depthVis, 0.5, color, 0.5, 0.0, combined);
		cv::imshow("combined", combined);
	}*/
	return 0;
}

int Rslam::testImu() {
	//rs2::pipeline pipe;
	//rs2::config cfg;
	//cfg.enable_stream(RS2_STREAM_ACCEL, RS2_FORMAT_MOTION_XYZ32F);
	//cfg.enable_stream(RS2_STREAM_GYRO, RS2_FORMAT_MOTION_XYZ32F);
	//pipe.start(cfg);

	//double last_ts[RS2_STREAM_COUNT];
	//double dt[RS2_STREAM_COUNT];

	//cv::Mat imDisp = cv::Mat::zeros(480, 848, CV_8UC3);
	//cv::Scalar textColor = CV_RGB(255, 255, 255);

	//while (cv::waitKey(1) < 0) {
	//	//frameset = pipe.wait_for_frames();
	//	if (!pipe.poll_for_frames(&frameset)) {
	//		continue;
	//	}

	//	for (auto f : frameset)
	//	{
	//		rs2::stream_profile profile = f.get_profile();

	//		unsigned long fnum = f.get_frame_number();
	//		double ts = f.get_timestamp();
	//		dt[profile.stream_type()] = (ts - last_ts[profile.stream_type()]) / 1000.0;
	//		last_ts[profile.stream_type()] = ts;

	//		/*std::cout << std::setprecision(12)
	//			<< "[ " << profile.stream_name()
	//			<< " ts: " << ts
	//			<< " dt: " << dt[profile.stream_type()]
	//			<< "] ";*/
	//	}

	//	auto fa = frameset.first(RS2_STREAM_ACCEL, RS2_FORMAT_MOTION_XYZ32F);
	//	rs2::motion_frame accel = fa.as<rs2::motion_frame>();

	//	auto fg = frameset.first(RS2_STREAM_GYRO, RS2_FORMAT_MOTION_XYZ32F);
	//	rs2::motion_frame gyro = fg.as<rs2::motion_frame>();

	//	rs2_vector gv = gyro.get_motion_data();
	//	double ratePitch = gv.x - GYRO_BIAS_X;
	//	double rateYaw = gv.y - GYRO_BIAS_Y;
	//	double rateRoll = gv.z - GYRO_BIAS_Z;

	//	if ((ratePitch > GYRO_MAX_X) || (ratePitch < GYRO_MIN_X)
	//		|| (rateYaw > GYRO_MAX_Y) || (rateYaw < GYRO_MIN_Y)
	//		|| (rateRoll > GYRO_MAX_Z) || (rateRoll < GYRO_MIN_Z)) {

	//		std::ostringstream message1, message2, message3;
	//		message1 << std::fixed << parseDecimal(rateRoll) << " "
	//			<< parseDecimal(ratePitch) << " "
	//			<< parseDecimal(rateYaw);
	//		imDisp.setTo(cv::Scalar((rateRoll + 10) * 20, (ratePitch + 10) * 20, (rateYaw + 10) * 20));
	//		cv::putText(imDisp, message1.str(), cv::Point(0, 10), cv::FONT_HERSHEY_PLAIN, 1, textColor);
	//		cv::imshow("imu", imDisp);
	//		//std::cout << "dt=" << dt[RS2_STREAM_GYRO]
	//		//	<< " rateRoll=" << rateRoll// * 180.0 / CV_PI 
	//		//	<< " ratePitch=" << ratePitch// * 180.0 / CV_PI 
	//		//	<< " rateYaw=" << rateYaw// * 180.0 / CV_PI 
	//		//	<< std::endl;
	//	}
	//}
	return 0;
}

int Rslam::showAlignedDepth() {
	try
	{
		rs2::config cfg;
		cfg.enable_stream(RS2_STREAM_DEPTH, 848, 480, rs2_format::RS2_FORMAT_Z16, 60);
		cfg.enable_stream(RS2_STREAM_COLOR, 848, 480, RS2_FORMAT_BGR8, 60);
		rs2::pipeline pipe;
		pipe.start(cfg);
		rs2::colorizer c;

		rs2::align align_to_color(RS2_STREAM_COLOR);

		const auto window_name = "Display Image";
		cv::namedWindow(window_name, cv::WINDOW_AUTOSIZE);

		rs2::frameset frameset0 = pipe.wait_for_frames();
		frameset0 = align_to_color.process(frameset0);
		rs2::frameset frameset1 = pipe.wait_for_frames();
		frameset1 = align_to_color.process(frameset1);
		rs2::frameset frameset2;

		while (cv::waitKey(1) < 0 && cv::getWindowProperty(window_name, cv::WND_PROP_AUTOSIZE) >= 0)
		{
			frameset2 = pipe.wait_for_frames();
			frameset2 = align_to_color.process(frameset2);
			auto depth = frameset0.get_depth_frame();
			auto color = frameset2.get_color_frame();
			auto colorized_depth = c.colorize(depth);

			frameset0 = frameset1;
			frameset1 = frameset2;


			// Query frame size (width and height)
			const int w = depth.as<rs2::video_frame>().get_width();
			const int h = depth.as<rs2::video_frame>().get_height();

			// Create OpenCV matrix of size (w,h) from the colorized depth data
			cv::Mat depthimage(cv::Size(w, h), CV_8UC3, (void*)colorized_depth.get_data(), cv::Mat::AUTO_STEP);
			cv::Mat colorimage(cv::Size(w, h), CV_8UC3, (void*)color.get_data(), cv::Mat::AUTO_STEP);

			// Update the window with new data
			cv::imshow(window_name, depthimage);
			cv::imshow("color", colorimage);

			cv::Mat combined;
			cv::addWeighted(depthimage, 0.5, colorimage, 0.5, 0.0, combined);
			cv::imshow("combined", combined);
		}
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
}

int Rslam::showDepth() {
	try {
		// Declare depth colorizer for pretty visualization of depth data
		rs2::colorizer color_map;
		rs2::config cfg;
		rs2::rates_printer printer;
		cfg.enable_stream(RS2_STREAM_DEPTH, 640, 480, rs2_format::RS2_FORMAT_Z16, 90);
		cfg.enable_stream(RS2_STREAM_INFRARED, 1, 640, 480, RS2_FORMAT_Y8, 90);
		cfg.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_BGR8, 60);
		//cfg.enable_stream(RS2_STREAM_INFRARED, 2, 640, 480, RS2_FORMAT_Y8, 90);

		// Declare RealSense pipeline, encapsulating the actual device and sensors
		rs2::pipeline pipe;
		// Start streaming with default recommended configuration
		pipe.start(cfg);

		const auto window_name = "Display Image";
		cv::namedWindow(window_name, cv::WINDOW_AUTOSIZE);

		while (cv::waitKey(1) < 0 && cv::getWindowProperty(window_name, cv::WND_PROP_AUTOSIZE) >= 0)
		{
			rs2::frameset data = pipe.wait_for_frames(); // Wait for next set of frames from the camera
			//std::cout << data.size() << std::endl;
			rs2::frame depth = data.get_depth_frame().apply_filter(color_map);
			rs2::frame infrared = data.get_infrared_frame();
			rs2::frame color = data.get_color_frame();

			// Query frame size (width and height)
			const int w = infrared.as<rs2::video_frame>().get_width();
			const int h = infrared.as<rs2::video_frame>().get_height();

			// Create OpenCV matrix of size (w,h) from the colorized depth data
			cv::Mat depthimage(cv::Size(w, h), CV_8UC3, (void*)depth.get_data(), cv::Mat::AUTO_STEP);
			cv::Mat irimage(cv::Size(w, h), CV_8UC1, (void*)infrared.get_data(), cv::Mat::AUTO_STEP);
			cv::Mat colorimage(cv::Size(w, h), CV_8UC3, (void*)color.get_data(), cv::Mat::AUTO_STEP);

			// Update the window with new data
			cv::imshow(window_name, depthimage);
			cv::imshow("ir", irimage);
			cv::imshow("color", colorimage);
		}

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
}

int Rslam::extractTimeStamps() {
	//gyro.lastTs = gyro.ts;
	//accel.lastTs = accel.ts;
	//for (auto f : frameset)
	//{
	//	rs2::stream_profile profile = f.get_profile();
	//	timestamps[profile.stream_type()] = f.get_timestamp();

	//	/*std::cout << std::fixed
	//	<< "[ " << profile.stream_name() << "(" << profile.stream_type() << ")"
	//	<< " ts: " << timestamps[profile.stream_type()]
	//	<< "] ";*/
	//}
	////std::cout << std::endl;
	//accel.ts = timestamps[RS2_STREAM_ACCEL];
	////std::cout << std::fixed << gyro.ts << std::endl;
	return 0;
}

