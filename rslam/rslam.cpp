#include "rslam.h"

int Rslam::initialize(int width, int height, int fps, double cx, double cy, double fx, double fy) {
	this->width = width;
	this->height = height;
	this->fps = fps;
	try {
		ctx = new rs2::context();
		pipe = new rs2::pipeline(*ctx);
		auto dev = ctx->query_devices();
		for (auto&& devfound : dev) {
			std::cout << "Found device: " << devfound.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER) << std::endl;
		}
		//std::cout << dev.size() << std::endl;
		//std::cout << "Found: " << dev[0].get_info(RS2_CAMERA_INFO_SERIAL_NUMBER) << std::endl;
		//std::cout << "Found: " << dev[1].get_info(RS2_CAMERA_INFO_SERIAL_NUMBER) << std::endl;
		rs2::config cfg;
		cfg.enable_device("843112071357");
		cfg.enable_stream(RS2_STREAM_DEPTH, this->width, this->height, rs2_format::RS2_FORMAT_Z16, this->fps);
		cfg.enable_stream(RS2_STREAM_COLOR, this->width, this->height, RS2_FORMAT_BGR8, this->fps);
		cfg.enable_stream(RS2_STREAM_ACCEL, RS2_FORMAT_MOTION_XYZ32F, 250);
		cfg.enable_stream(RS2_STREAM_GYRO, RS2_FORMAT_MOTION_XYZ32F, 400);
		pipe->start(cfg);
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

	depth = cv::Mat(this->height, this->width, CV_16S);
	depthVis = cv::Mat(this->height, this->width, CV_8UC3);
	color = cv::Mat(this->height, this->width, CV_8UC3);

	/*camerapose = new CameraPose();
	double intrinsicData[9] = { fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0 };
	intrinsic = cv::Mat(3, 3, CV_64F, intrinsicData).clone();
	camerapose->initialize(intrinsic);*/

	// Utilities
	gyroDisp = cv::Mat::zeros(200, 600, CV_8UC3);
	accelDisp = cv::Mat::zeros(200, 600, CV_8UC3);

	// SLAM
	minHessian = 10000;
	surf = cv::cuda::SURF_CUDA(this->minHessian);
	surf.hessianThreshold = minHessian;
	matcher = cv::cuda::DescriptorMatcher::createBFMatcher();

	return EXIT_SUCCESS;
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
	else { //load default
		fx = 614.122;
		fy = 614.365;
		cx = 427.388;
		cy = 238.478;
		width = 640;
		height = 480;
		fps = 60;
	}
	return initialize(width, height, fps, cx, cy, fx, fy);
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
	double duration;
	// Start pipes as recorders
	for (auto&& dev : context.query_devices())
	{
		rs2::pipeline pipe(context);
		rs2::config cfg;
		cfg.enable_device(dev.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER));
		cfg.enable_record_to_file(dev.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER));
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
		if (pressed == 27) break; //press escape

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
		if ((pressed == 'g') || ((std::clock() - start) / (double)CLOCKS_PER_SEC > 10.0)) {
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
	std::thread t1(&Rslam::visualizePose, this);
	std::thread t2(&Rslam::poseSolver, this);
	t1.join();
	t2.join();
}

// Main loop for pose estimation
int Rslam::poseSolver() {
	gyro.ts = 0.0;
	accel.ts = 0.0;
	double last_ts[RS2_STREAM_COUNT];
	double dt[RS2_STREAM_COUNT];
	while (cv::waitKey(1) < 0) {
		if (!pipe->poll_for_frames(&frameset)) {
			continue;
		}
		frameset = alignToColor.process(frameset);
		//extractTimeStamps();
		extractGyroAndAccel();
		visualizeImu();
		updatePose();

		// Get color and depth frames
		extractColorAndDepth();

		//visualizeColor();
		//visualizeDepth();
		solveKeypointsAndDescriptors(color);
		visualizeKeypoints();

		/*std::cout << std::fixed
			<< gyro.ts << " " << gyro.lastTs << " "
			<< gyro.dt << ": ("
			<< gyro.x << ","
			<< gyro.y << ","
			<< gyro.z << " )"
			<< accel.dt << ": ("
			<< accel.x << " "
			<< accel.y << " "
			<< accel.z << ")"
			<< std::endl;*/
	}
	// poll for frames (gyro, accel)
	// if depth and image available, fetch depth and image
	// set first stable frame as reference
	// align depth with image
	// feature extraction
	// feature matching
	// pose estimation
	return 0;
}

int Rslam::solveKeypointsAndDescriptors(cv::Mat im) {
	cv::Mat gray;
	cv::cvtColor(im, gray, CV_BGR2GRAY);
	d_im.upload(gray); //first cuda call is always slow
	surf(d_im, cv::cuda::GpuMat(), d_keypoints, d_descriptors);
	surf.downloadKeypoints(d_keypoints, keypoints);
	return 0;
}

int Rslam::extractGyroAndAccel() {
	gyro.lastTs = gyro.ts;
	auto gyroFrame = frameset.first(RS2_STREAM_GYRO, RS2_FORMAT_MOTION_XYZ32F).as<rs2::motion_frame>();
	rs2_vector gv = gyroFrame.get_motion_data();
	gyro.x = gv.x - GYRO_BIAS_X;
	gyro.y = gv.y - GYRO_BIAS_Y;
	gyro.z = gv.z - GYRO_BIAS_Z;
	gyro.ts = gyroFrame.get_timestamp();
	gyro.dt = (gyro.ts - gyro.lastTs) / 1000.0;

	accel.lastTs = accel.ts;
	auto accelFrame = frameset.first(RS2_STREAM_ACCEL, RS2_FORMAT_MOTION_XYZ32F).as<rs2::motion_frame>();
	rs2_vector av = accelFrame.get_motion_data();
	accel.x = av.x;
	accel.y = av.y;
	accel.z = av.z;
	accel.ts = accelFrame.get_timestamp();
	accel.dt = (accel.ts - accel.lastTs) / 1000.0;
	//float R = sqrtf(av.x * av.x + av.y * av.y + av.z * av.z);
	//float newRoll = acos(av.x / R);
	//float newYaw = acos(av.y / R);
	//float newPitch = acos(av.z / R);
	//std::cout << accel.dt << std::endl;
	return 0;
}

int Rslam::extractColorAndDepth() {
	auto colorData = frameset.get_color_frame();
	color = cv::Mat(cv::Size(width, height), CV_8UC3, (void*)colorData.get_data(), cv::Mat::AUTO_STEP);
	auto depthData = frameset.get_depth_frame();
	depth = cv::Mat(cv::Size(width, height), CV_16S, (void*)depthData.get_data(), cv::Mat::AUTO_STEP);
	return 0;
}

void Rslam::updatePose() {
	pose.x = accel.x;
	pose.y = accel.y;
	pose.z = accel.z;
	pose.rx = 0.0f;
	pose.ry = 0.0f;
	pose.rz = 0.0f;
	pose.rw = 1.0f;
}

// Utilities
void Rslam::visualizeImu() {
	std::ostringstream gyroValx, gyroValy, gyroValz;
	gyroValx << std::fixed << parseDecimal(gyro.x);
	gyroValy << std::fixed << parseDecimal(gyro.y);
	gyroValz << std::fixed << parseDecimal(gyro.z);
	//gyroDisp.setTo(cv::Scalar((gyro.x + 10) * 20, (gyro.y + 10) * 20, (gyro.z + 10) * 20));
	gyroDisp.setTo(cv::Scalar(50, 50, 50));
	cv::circle(gyroDisp, cv::Point(100, 100), abs(10.0*gyro.x), cv::Scalar(0, 0, 255), -1);
	cv::circle(gyroDisp, cv::Point(300, 100), abs(10.0*gyro.y), cv::Scalar(0, 255, 0), -1);
	cv::circle(gyroDisp, cv::Point(500, 100), abs(10.0*gyro.z), cv::Scalar(255, 0, 0), -1);

	cv::putText(gyroDisp, gyroValx.str(), cv::Point(0, 10), cv::FONT_HERSHEY_PLAIN, 1, CV_RGB(255, 255, 255));
	cv::putText(gyroDisp, gyroValy.str(), cv::Point(200, 10), cv::FONT_HERSHEY_PLAIN, 1, CV_RGB(255, 255, 255));
	cv::putText(gyroDisp, gyroValz.str(), cv::Point(400, 10), cv::FONT_HERSHEY_PLAIN, 1, CV_RGB(255, 255, 255));
	cv::imshow("gyro", gyroDisp);

	std::ostringstream accelValx, accelValy, accelValz;
	accelValx << std::fixed << parseDecimal(accel.x);
	accelValy << std::fixed << parseDecimal(accel.y);
	accelValz << std::fixed << parseDecimal(accel.z);
	//gyroDisp.setTo(cv::Scalar((gyro.x + 10) * 20, (gyro.y + 10) * 20, (gyro.z + 10) * 20));
	accelDisp.setTo(cv::Scalar(50, 50, 50));
	cv::circle(accelDisp, cv::Point(100, 100), abs(5.0*accel.x), cv::Scalar(0, 0, 255), -1);
	cv::circle(accelDisp, cv::Point(300, 100), abs(5.0*accel.y), cv::Scalar(0, 255, 0), -1);
	cv::circle(accelDisp, cv::Point(500, 100), abs(5.0*accel.z), cv::Scalar(255, 0, 0), -1);

	cv::putText(accelDisp, accelValx.str(), cv::Point(0, 10), cv::FONT_HERSHEY_PLAIN, 1, CV_RGB(255, 255, 255));
	cv::putText(accelDisp, accelValy.str(), cv::Point(200, 10), cv::FONT_HERSHEY_PLAIN, 1, CV_RGB(255, 255, 255));
	cv::putText(accelDisp, accelValz.str(), cv::Point(400, 10), cv::FONT_HERSHEY_PLAIN, 1, CV_RGB(255, 255, 255));
	cv::imshow("accel", accelDisp);
}

void Rslam::visualizePose() {
	Viewer *viewer = new Viewer();
	viewer->createWindow(800, 600, "pose");
	viewer->setCameraProjectionType(Viewer::ProjectionType::PERSPECTIVE);

	FileReader *objFile = new FileReader();
	objFile->readObj("D:/dev/c-projects/OpenSor/test_opensor_viewer/data/models/monkey.obj", FileReader::ArrayFormat::VERTEX_NORMAL_TEXTURE, 0.01f);

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
}

void Rslam::visualizeColor() {
	//auto colorData = frameset.get_color_frame();
	//color = cv::Mat(cv::Size(width, height), CV_8UC3, (void*)colorData.get_data(), cv::Mat::AUTO_STEP);
	cv::imshow("color", color);
}

void Rslam::visualizeDepth() {
	auto depthData = frameset.get_depth_frame();
	auto depthVisData = colorizer.colorize(depthData);
	depthVis = cv::Mat(cv::Size(width, height), CV_8UC3, (void*)depthVisData.get_data(), cv::Mat::AUTO_STEP);
	cv::imshow("depth", depthVis);
}

void Rslam::visualizeKeypoints() {
	cv::Mat imout;
	cv::drawKeypoints(color, keypoints, imout, cv::Scalar(0, 255, 0), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	cv::imshow("keypoints", imout);
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

























// TESTS **********************************************
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
	rs2::pipeline pipe;
	frameset = pipe.wait_for_frames();
	frameset = alignToColor.process(frameset);
	auto depthData = frameset.get_depth_frame();
	auto colorData = frameset.get_color_frame();
	auto depthVisData = colorizer.colorize(depthData);
	depthVis = cv::Mat(cv::Size(width, height), CV_8UC3, (void*)depthVisData.get_data(), cv::Mat::AUTO_STEP);
	color = cv::Mat(cv::Size(width, height), CV_8UC3, (void*)colorData.get_data(), cv::Mat::AUTO_STEP);
	return 0;
}

int Rslam::getPose(float *x, float *y, float *z, float *roll, float *pitch, float *yaw) {
	return 0;
}

int Rslam::getGyro(float *rateRoll, float *ratePitch, float *rateYaw) {
	rs2::pipeline pipe;
	frameset = pipe.wait_for_frames();
	rs2_vector gv = frameset.first(RS2_STREAM_GYRO, RS2_FORMAT_MOTION_XYZ32F).as<rs2::motion_frame>().get_motion_data();
	gyro.x = gv.x - GYRO_BIAS_X;
	gyro.y = gv.y - GYRO_BIAS_Y;
	gyro.z = gv.z - GYRO_BIAS_Z;
	*rateRoll = gyro.x;
	*ratePitch = gyro.y;
	*rateYaw = gyro.z;
	//std::cout << gyroRatePitch << std::endl;
	return 0;
}

int Rslam::testStream() {
	while (cv::waitKey(1) < 0)
	{
		getFrames();
		cv::imshow("depth", depthVis);
		cv::imshow("color", color);
		cv::Mat combined;
		cv::addWeighted(depthVis, 0.5, color, 0.5, 0.0, combined);
		cv::imshow("combined", combined);
	}
	return 0;
}

int Rslam::testImu() {
	rs2::pipeline pipe;
	rs2::config cfg;
	cfg.enable_stream(RS2_STREAM_ACCEL, RS2_FORMAT_MOTION_XYZ32F);
	cfg.enable_stream(RS2_STREAM_GYRO, RS2_FORMAT_MOTION_XYZ32F);
	pipe.start(cfg);

	double last_ts[RS2_STREAM_COUNT];
	double dt[RS2_STREAM_COUNT];

	cv::Mat imDisp = cv::Mat::zeros(480, 848, CV_8UC3);
	cv::Scalar textColor = CV_RGB(255, 255, 255);

	while (cv::waitKey(1) < 0) {
		//frameset = pipe.wait_for_frames();
		if (!pipe.poll_for_frames(&frameset)) {
			continue;
		}

		for (auto f : frameset)
		{
			rs2::stream_profile profile = f.get_profile();

			unsigned long fnum = f.get_frame_number();
			double ts = f.get_timestamp();
			dt[profile.stream_type()] = (ts - last_ts[profile.stream_type()]) / 1000.0;
			last_ts[profile.stream_type()] = ts;

			/*std::cout << std::setprecision(12)
				<< "[ " << profile.stream_name()
				<< " ts: " << ts
				<< " dt: " << dt[profile.stream_type()]
				<< "] ";*/
		}

		auto fa = frameset.first(RS2_STREAM_ACCEL, RS2_FORMAT_MOTION_XYZ32F);
		rs2::motion_frame accel = fa.as<rs2::motion_frame>();

		auto fg = frameset.first(RS2_STREAM_GYRO, RS2_FORMAT_MOTION_XYZ32F);
		rs2::motion_frame gyro = fg.as<rs2::motion_frame>();

		rs2_vector gv = gyro.get_motion_data();
		double ratePitch = gv.x - GYRO_BIAS_X;
		double rateYaw = gv.y - GYRO_BIAS_Y;
		double rateRoll = gv.z - GYRO_BIAS_Z;

		if ((ratePitch > GYRO_MAX_X) || (ratePitch < GYRO_MIN_X)
			|| (rateYaw > GYRO_MAX_Y) || (rateYaw < GYRO_MIN_Y)
			|| (rateRoll > GYRO_MAX_Z) || (rateRoll < GYRO_MIN_Z)) {

			std::ostringstream message1, message2, message3;
			message1 << std::fixed << parseDecimal(rateRoll) << " "
				<< parseDecimal(ratePitch) << " "
				<< parseDecimal(rateYaw);
			imDisp.setTo(cv::Scalar((rateRoll + 10) * 20, (ratePitch + 10) * 20, (rateYaw + 10) * 20));
			cv::putText(imDisp, message1.str(), cv::Point(0, 10), cv::FONT_HERSHEY_PLAIN, 1, textColor);
			cv::imshow("imu", imDisp);
			//std::cout << "dt=" << dt[RS2_STREAM_GYRO]
			//	<< " rateRoll=" << rateRoll// * 180.0 / CV_PI 
			//	<< " ratePitch=" << ratePitch// * 180.0 / CV_PI 
			//	<< " rateYaw=" << rateYaw// * 180.0 / CV_PI 
			//	<< std::endl;
		}
	}
	return 0;
}



int Rslam::solveRelativePose() {


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
	gyro.lastTs = gyro.ts;
	accel.lastTs = accel.ts;
	for (auto f : frameset)
	{
		rs2::stream_profile profile = f.get_profile();
		timestamps[profile.stream_type()] = f.get_timestamp();

		/*std::cout << std::fixed
		<< "[ " << profile.stream_name() << "(" << profile.stream_type() << ")"
		<< " ts: " << timestamps[profile.stream_type()]
		<< "] ";*/
	}
	//std::cout << std::endl;
	accel.ts = timestamps[RS2_STREAM_ACCEL];
	//std::cout << std::fixed << gyro.ts << std::endl;
	return 0;
}

