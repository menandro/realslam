/*
Vehicle Localization from Driver-Mounted Stereo Fisheye Cameras
Device: T265
*/
#pragma once

#include <librealsense2\rs.hpp>
#include <opencv2/opencv.hpp>
#include "lib_link.h"
#include <viewer\Viewer.h>
#include <filereader\Filereader.h>
#include <upsampling/upsampling.h>
#include <thread>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>
#include <opencv2/xfeatures2d/cuda.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/optflow.hpp>
#include <stereo/stereo.h>
#include <stereotgv/stereotgv.h>
#include <stereolite/stereolite.h>

class Tslam {
public:
	Tslam() {};
	~Tslam() {};

	class Gyro {
	public:
		float x; //rate(dot) of rotation in x(rx)
		float y;
		float z;
		double ts; //timestamp
		double lastTs;
		double dt;
	};

	class Accel {
	public:
		float x;
		float y;
		float z;
		double ts; //timestamp
		double lastTs;
		double dt;
	};

	class Quaternion {
	public:
		Quaternion() {};
		Quaternion(float x, float y, float z, float w) {
			this->x = x;
			this->y = y;
			this->z = z;
			this->w = w;
		}
		float x;
		float y;
		float z;
		float w;
	};

	class Vector3 {
	public:
		Vector3() {};
		Vector3(float x, float y, float z) {
			this->x = x;
			this->y = y;
			this->z = z;
		};
		float x;
		float y;
		float z;
	};

	class Keyframe {
	public:
		Keyframe() {};
		~Keyframe() {};
		cv::Mat im;
		cv::cuda::GpuMat d_im;
		std::vector<cv::KeyPoint> keypoints;
		cv::Mat descriptors;
		cv::cuda::GpuMat d_descriptors;

		// Keyframe global pose
		cv::Mat depth;
		cv::Mat R;
		cv::Mat t;

		// Current frame relative pose
		cv::Mat currentRelativeR;
		cv::Mat currentRelativeT;
		cv::Mat currentGlobalR;
		cv::Mat currentGlobalT;

		// These values change every frame
		std::vector<cv::KeyPoint> matchedKeypoints;
		std::vector<cv::KeyPoint> matchedKeypointsSrc;
		std::vector<cv::Point2f> matchedPoints;
		std::vector<cv::Point2f> matchedPointsSrc;
		std::vector<cv::Point3f> objectPointsSrc;
		std::vector<float> matchedDistances;
	};

	// T265
	class T265 {
	public:
		T265() {};
		~T265() {};

		bool isFound = false;
		std::string serialNo;

		rs2::pipeline * pipe;
		rs2::frameset frameset;
		rs2::config cfg;

		const int width = 848;
		const int height = 800;

		cv::Mat fisheye1;
		cv::Mat fisheye2;
		cv::Mat fisheye132f;
		cv::Mat fisheye232f;
		cv::Mat fisheyeMask;
		cv::Mat depth32f = cv::Mat(height, width, CV_32F);
		cv::Mat depthHalf32f = cv::Mat(height / 2, width / 2, CV_32F);
		cv::Mat mask;

		cv::cuda::GpuMat d_fe1;
		cv::cuda::GpuMat d_fe2;
		cv::cuda::GpuMat d_fisheyeMask;
		cv::cuda::GpuMat d_keypointsFe1;
		cv::cuda::GpuMat d_descriptorsFe1;
		cv::cuda::GpuMat d_keypointsFe2;
		cv::cuda::GpuMat d_descriptorsFe2;
		std::vector<cv::KeyPoint> keypointsFe1;
		std::vector<cv::KeyPoint> keypointsFe2;
		cv::Mat descriptorsFe1;
		cv::Mat descriptorsFe2;

		// Stereo Matching
		std::vector<cv::KeyPoint> stereoKeypoints;
		std::vector<cv::KeyPoint> stereoKeypointsSrc;
		std::vector<cv::Point2f> stereoPoints;
		std::vector<cv::Point2f> stereoPointsSrc;
		std::vector<cv::Point3f> stereoObjectPointsSrc;
		std::vector<float> stereoDistances;
		std::vector<std::vector<cv::DMatch>> stereoMatches;

		// Keyframe Matching
		std::vector<std::vector<cv::DMatch>> keyframeMatches;

		rs2::frame_queue gyroQueue = rs2::frame_queue(1);
		rs2::frame_queue accelQueue = rs2::frame_queue(1);

		Gyro gyro;
		Accel accel;
		Quaternion ImuRotation;
		Vector3 ImuTranslation;
		Vector3 ImuVelocity;

		Keyframe * currentKeyframe;
		bool keyframeExist;
	};

	

	// Device
	rs2::context * ctx;
	T265 t265;

	// Viewer
	Viewer* viewer;

	// Stereo
	Stereo* stereo;
	StereoTgv* stereotgv;
	StereoLite* stereolite;
	int stereoWidth;
	int stereoHeight;
	float stereoScaling;
	cv::Mat fisheyeMask;
	cv::Mat fisheyeMask8;
	cv::Mat fisheyeMaskHalf;
	cv::Mat fisheyeMask8Half;

	// Upsampling
	lup::Upsampling * upsampling;
	float maxUpsamplingDepth;

	// SLAM
	cv::Ptr<cv::cuda::ORB> orb;
	cv::Ptr< cv::cuda::DescriptorMatcher > matcher;

	int initialize(const char* serialNumber);
	int run();

	// Threads
	int fetchFrames();
	int imuPoseSolver();
	int cameraPoseSolver();
	int visualizePose();

	int detectAndComputeOrb(cv::Mat im, cv::cuda::GpuMat &d_im,
		std::vector<cv::KeyPoint> &keypoints, cv::cuda::GpuMat &descriptors);
	//int detectAndComputeOrb(T265 &device);
	int matchAndPose(T265& device);
	int stereoMatching(T265 &device);

	bool settleImu(T265 &device);
	int solveImuPose(T265 &device);
	void updateViewerImuPose(T265 &device);

	bool processGyro(T265 &device);
	bool processAccel(T265 &device);

	// Stereo
	int initStereoTVL1();
	int initStereoTGVL1();
	int solveStereoTGVL1();
	int solveStereoTVL1();
	int initDepthUpsampling();
	cv::Mat depthVisMask;
	bool isDepthVisMaskCreated = false;


	// Utilities
	int createDepthThresholdMask(T265 &device, float maxDepth);
	void visualizeKeypoints(T265 &device, std::string windowNamePrefix);
	void visualizeMatchedStereoPoints(T265 &device, std::string windowNamePrefix);
	void visualizeRelativeKeypoints(Keyframe *keyframe, cv::Mat ir1, std::string windowNamePrefix);
	bool isThisDevice(std::string serialNo, std::string queryNo);
	void overlayMatrixRot(const char* windowName, cv::Mat& im, Vector3 euler, Quaternion q);
	std::string parseDecimal(double f);
	std::string parseDecimal(double f, int precision);

	void testStereo(std::string im1, std::string im2);
	void showDepthJet(std::string windowName, cv::Mat image, float maxDepth, bool shouldWait);
	void showDepthJet(std::string windowName, cv::Mat image, std::string message, float maxDepth, bool shouldWait);
	int saveT265Images(std::string filename, std::string folderOutput);
};