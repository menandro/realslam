#pragma once

#define GLM_ENABLE_EXPERIMENTAL

#include <librealsense2\rs.hpp>
#include <opencv2/opencv.hpp>
#include "lib_link.h"
#include <thread>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>
#include <opencv2/xfeatures2d/cuda.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/optflow.hpp>
#include <stereo/stereo.h>
#include <stereotgv/stereotgv.h>
#include <stereolite/stereolite.h>
#include <opticalflow/flow.h>
#include <imu/imu.h>
#include <atomic>

// Outputs Images, Depth Map, 3D points and IMU
class T265 {
public:
	T265();
	~T265() {};

	rs2::context* ctx;

	bool isFound = false;
	std::string serialNo;

	rs2::pipeline* pipe;
	rs2::frameset frameset;
	rs2::config cfg;

	const int width = 848;
	const int height = 800;
	cv::Mat intrinsic;
	cv::Mat distCoeffs;

	// Output
	cv::Mat image;
	cv::Mat X;

	// Data
	cv::Mat fisheye1;
	cv::Mat fisheye2;
	cv::Mat fisheye132f;
	cv::Mat fisheye232f;
	cv::Mat fisheyeMask;
	cv::Mat fisheyeMask8;
	cv::Mat fisheye1texture;
	cv::Mat fisheye28uc3;
	cv::Mat mask;

	// Output Dense Stereo
	cv::Mat depth32f = cv::Mat(height, width, CV_32F);
	cv::Mat depthHalf32f = cv::Mat(height / 2, width / 2, CV_32F);
	cv::Mat Xraw; // 3D points

	// Output IMU
	Quaternion ImuRotation;
	Vector3 ImuTranslation; // Not used
	Vector3 ImuVelocity; // Not used

	rs2::frame_queue gyroQueue = rs2::frame_queue(1);
	rs2::frame_queue accelQueue = rs2::frame_queue(1);

	Gyro gyro;
	Accel accel;
	
	// Stereo
	StereoTgv* stereotgv;
	int stereoWidth;;
	int stereoHeight;
	float stereoScaling = 1.0f;
	int initStereoTGVL1();
	int solveStereoTGVL1();

	// Functions
	int initialize(const char* serialNumber, float stereoScaling);
	int initialize(const char* serialNumber);
	int run();
	int stop();
	int stopThreadCheck();
	int fetchFrames();
	int imuPoseSolver();
	std::atomic<bool> isFetchFramesRunning = false;
	std::atomic<bool> isImuPoseSolverRunning = false;

	// IMU
	bool settleImu();
	int solveImuPose();
	bool processGyro();
	bool processAccel();

	// Utilities
	bool isThisDevice(std::string serialNo, std::string queryNo);
	void showDepthJet(std::string windowName, cv::Mat image, float maxDepth, bool shouldWait);
	void showDepthJet(std::string windowName, cv::Mat image, std::string message,
		float maxDepth, bool shouldWait);
	
};