#include "fisheyeslam.h"

FisheyeSlam::FisheyeSlam() {

}

int FisheyeSlam::initialize(int width, int height, cv::Mat intrinsic, cv::Mat distCoeffs, cv::Mat fisheyeMask8uc1) {
	this->height = height;
	this->width = width;
	this->fisheyeMask8uc1 = fisheyeMask8uc1.clone();
	fisheyeMask8uc1.convertTo(fisheyeMask32fc1, CV_32F, 1.0 / 255.0);

	viewer = new Viewer();
	keyframes = std::vector<Keyframe>();
	currKFIndex = 0;
	this->intrinsic = intrinsic;
	this->distortionCoeffs = distCoeffs;
	initOpticalFlow();

	return 0;
}

int FisheyeSlam::updateImageSlam(cv::Mat im, cv::Mat point3d) {
	currImage = im;
	currX = point3d;
	point3d.reshape(3, point3d.rows * point3d.cols).copyTo(currObjectPoints);
	cv::imshow("sdfef", point3d);
	isImageUpdated = true;
	return 0;
}

int FisheyeSlam::updateImu(Quaternion imuRotation) {
	this->imuRotation = imuRotation;
	isImuUpdated = true;
	return 0;
}

int FisheyeSlam::run() {
	std::thread t0(&FisheyeSlam::tracking, this);
	std::thread t1(&FisheyeSlam::visualizePose, this);
	std::thread t2(&FisheyeSlam::mapping, this);
	t0.detach();
	t1.detach();
	t2.detach();
	return 0;
}

int FisheyeSlam::tracking() {
	std::this_thread::sleep_for(std::chrono::seconds(5));

	cv::Mat im = cv::Mat::zeros(100, 400, CV_8UC3);
	cv::putText(im, "Tracking Thread", cv::Point(0, 30), cv::FONT_HERSHEY_PLAIN, 2, CV_RGB(255, 255, 255));
	cv::imshow("Tracking Thread", im);
	while (1) {
		char pressed = cv::waitKey(10);
		if (pressed == 27) break;

		if (!isImageUpdated) continue;

		if (!keyframeExist) { // Change this to flag that can be updated
			Keyframe kf;
			kf.im = currImage.clone();
			kf.X = currX.clone();
			kf.objectPoints = currObjectPoints;
			keyframes.push_back(kf);
			currKFIndex = 0;
			keyframeExist = true;
		}
		else {
			// Track with keyframe
			//cv::imshow("im1", currImage);
			//cv::imshow("im2", prevImage);
			solveOpticalFlow(prevImage, currImage);
			prevImage = currImage.clone();

			// Solve PnP - For Fisheye (no undistortion)
			auto kf = keyframes[currKFIndex];
			cv::imshow("kf", kf.X);
			try {
				//std::cout << kf.objectPoints.size() << " " << currMatchedPoints.size() << std::endl;
				cv::solvePnPRansac(kf.objectPoints, currMatchedPoints,
					this->intrinsic, this->distortionCoeffs, kf.Rrel, kf.trel, false);
			}
			catch (const std::exception & e)
			{
				std::cerr << "Insufficient Points" << std::endl;
				continue;
				//std::cerr << e.what() << std::endl;
			}
			cv::Mat im = cv::Mat::zeros(100, 300, CV_8UC3);
			im.setTo(cv::Scalar(50, 50, 50));
			overlayMatrix("pose", im, kf.Rrel, kf.trel);
			//updateViewerCameraPose(kf.R, kf.t);
		}
		isImageUpdated = false;
	}
	
	return 0;
}

int FisheyeSlam::mapping() {
	while (true) {
		if (!isImuUpdated) continue;

		// Update Visualize Pose
		updateViewerImuPose();

		isImuUpdated = false;
	}
	return 0;
}

int FisheyeSlam::visualizePose() {
	viewer->createWindow(800, 600, "campose");
	viewer->setCameraProjectionType(Viewer::ProjectionType::PERSPECTIVE);

	FileReader* objFile = new FileReader();
	float scale = 0.03f;
	objFile->readObj("monkey.obj", FileReader::ArrayFormat::VERTEX_NORMAL_TEXTURE, scale);
	FileReader* floorFile = new FileReader();
	floorFile->readObj("floor.obj", FileReader::ArrayFormat::VERTEX_NORMAL_TEXTURE, scale);
	FileReader* viewBoxFile = new FileReader();
	viewBoxFile->readObj("view.obj", FileReader::ArrayFormat::VERTEX_NORMAL_TEXTURE, scale);

	//Axis for Camera pose
	//cv::Mat texture = cv::imread("texture.png");
	CgObject* camAxis = new CgObject();
	camAxis->loadShader("myshader2.vert", "myshader2.frag");
	camAxis->loadData(objFile->vertexArray, objFile->indexArray, CgObject::ArrayFormat::VERTEX_NORMAL_TEXTURE);
	camAxis->loadTexture("monkey1.png");
	camAxis->setDrawMode(CgObject::Mode::TRIANGLES);
	camAxis->setLight();
	camAxis->objectIndex = (int)viewer->cgObject->size();
	viewer->cgObject->push_back(camAxis);

	// View box for Camera Pose
	CgObject* viewBox = new CgObject();
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
	CgObject* imuAxis = new CgObject();
	imuAxis->loadShader("myshader2.vert", "myshader2.frag");
	imuAxis->loadData(objFile->vertexArray, objFile->indexArray, CgObject::ArrayFormat::VERTEX_NORMAL_TEXTURE);
	imuAxis->loadTexture("monkey2.png");
	imuAxis->setDrawMode(CgObject::Mode::TRIANGLES);
	imuAxis->setLight();
	imuAxis->objectIndex = (int)viewer->cgObject->size();
	viewer->cgObject->push_back(imuAxis);

	CgObject* imuViewBox = new CgObject();
	imuViewBox->loadShader("edgeshader.vert", "edgeshader.frag");
	imuViewBox->loadData(viewBoxFile->vertexArray, viewBoxFile->indexArray, CgObject::ArrayFormat::VERTEX_NORMAL_TEXTURE);
	imuViewBox->loadTexture("default_texture.jpg");
	imuViewBox->setDrawMode(CgObject::Mode::CULLED_POINTS);
	imuViewBox->setLight();
	imuViewBox->setColor(cv::Scalar(1.0, 0, 0.0));
	imuViewBox->objectIndex = (int)viewer->cgObject->size();
	viewer->cgObject->push_back(imuViewBox);

	CgObject* floor = new CgObject();
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

void FisheyeSlam::updateViewerImuPose() {
	if (viewer->isRunning) {
		float translationScale = 1.0f;

		// update imuaxis
		glm::quat imuAdjust = glm::quat(glm::vec3(-1.57079632679f, 0.0f, 0.0f));
		glm::quat imuQ(this->imuRotation.w, this->imuRotation.x, this->imuRotation.y, this->imuRotation.z);
		glm::quat imuQRot = imuAdjust * imuQ;
		glm::vec3 euler = glm::eulerAngles(imuQRot);

		viewer->cgObject->at(2)->qrot = imuQRot;
		viewer->cgObject->at(3)->qrot = imuQRot;
	}
}

void FisheyeSlam::updateViewerCameraPose(cv::Mat Rvec, cv::Mat t) {
	if (viewer->isRunning) {
		float translationScale = 1.0f;
		//update camaxis

		glm::quat q(glm::vec3(-(float)Rvec.at<double>(0), -(float)Rvec.at<double>(1), (float)Rvec.at<double>(2)));
		viewer->cgObject->at(0)->qrot = q;

		/*viewer->cgObject->at(0)->tx = -(float)t.at<double>(0) * translationScale;
		viewer->cgObject->at(0)->ty = -(float)t.at<double>(1) * translationScale;
		viewer->cgObject->at(0)->tz = (float)t.at<double>(2) * translationScale;*/

		viewer->cgObject->at(1)->qrot = q;

		/*viewer->cgObject->at(1)->tx = -(float)t.at<double>(0) * translationScale;
		viewer->cgObject->at(1)->ty = -(float)t.at<double>(1) * translationScale;
		viewer->cgObject->at(1)->tz = (float)t.at<double>(2) * translationScale;*/
	}
}


void FisheyeSlam::overlayMatrix(const char* windowName, cv::Mat& im, cv::Mat R1, cv::Mat t) {
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

std::string FisheyeSlam::parseDecimal(double f) {
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

std::string FisheyeSlam::parseDecimal(double f, int precision) {
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