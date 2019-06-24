#pragma once
#include <librealsense2\rs.hpp>
#include <camerapose\CameraPose.h>
#include <opencv2/opencv.hpp>
#include "lib_link.h"
#include <viewer\Viewer.h>
#include <filereader\Filereader.h>
#include <upsampling/upsampling.h>
#include <thread>

class Rslam {
public:
	Rslam() {};
	~Rslam() {};

	// Global Pose
	cv::Mat Rvec;
	cv::Mat t;

	enum Settings {
		D435I_848_480_60,
		D435I_640_480_60,
		D435I_IR_640_360_90,
		T265
	};

	enum FeatureDetectionMethod {
		SURF,
		ORB
	};

	float GYRO_BIAS_X = 0.0011f;
	float GYRO_BIAS_Y = 0.0030f;
	float GYRO_BIAS_Z = -0.000371f;
	float GYRO_MAX10_X = 0.0716f;
	float GYRO_MAX10_Y = 0.0785f;
	float GYRO_MAX10_Z = 0.0873f;
	float GYRO_MIN10_X = -0.0698f;
	float GYRO_MIN10_Y = -0.0751f;
	float GYRO_MIN10_Z = -0.0890f;

	float GYRO_MAX5_X = 0.0960f;
	float GYRO_MAX5_Y = 0.1012f;
	float GYRO_MAX5_Z = 0.1152f;
	float GYRO_MIN5_X = -0.0890f;
	float GYRO_MIN5_Y = -0.0961f;
	float GYRO_MIN5_Z = -0.1152f;

	float GYRO_MAX_X = 0.2182f;
	float GYRO_MAX_Y = 0.2007f;
	float GYRO_MAX_Z = 0.2531f;
	float GYRO_MIN_X = -0.2147f;
	float GYRO_MIN_Y = -0.2200f;
	float GYRO_MIN_Z = -0.2496f;

	double timestamps[RS2_STREAM_COUNT];
	struct Gyro {
		float x; //rate(dot) of rotation in x(rx)
		float y;
		float z;
		double ts; //timestamp
		double lastTs;
		double dt;
	};

	struct Accel {
		float x;
		float y;
		float z;
		double ts; //timestamp
		double lastTs;
		double dt;
	};

	struct Pose {
		float x;
		float y;
		float z;
		float rx;
		float ry;
		float rz;
		float rw;
	};

	rs2::context * ctx;

	// For keyframing
	class Keyframe {
	public:
		Keyframe() {};
		~Keyframe() {};
		cv::Mat im;
		cv::cuda::GpuMat d_im;
		std::vector<cv::KeyPoint> keypoints;
		cv::Mat descriptors;
		cv::cuda::GpuMat d_descriptors;

		cv::Mat R;
		cv::Mat t;

		// These values change every frame
		std::vector<cv::KeyPoint> matchedKeypoints;
		std::vector<cv::KeyPoint> matchedKeypointsSrc;
		std::vector<cv::Point2f> matchedPoints;
		std::vector<cv::Point2f> matchedPointsSrc;
		std::vector<cv::Point3f> objectPointsSrc;
		std::vector<float> matchedDistances;
	};

	class Device {
	public:
		Device() {};
		~Device() {};
		std::string id;

		rs2::pipeline * pipe;
		rs2::frameset frameset;
		cv::Mat depth;
		cv::Mat depth32f;
		cv::Mat color;
		cv::Mat depthVis;
		cv::Mat infrared1;
		cv::Mat infrared2;
		cv::Mat infrared132f;
		cv::Mat infrared232f;

		// For pose estimation
		cv::cuda::GpuMat d_im;
		cv::cuda::GpuMat d_ir1;
		cv::cuda::GpuMat d_ir2;
		cv::cuda::GpuMat d_keypoints;
		cv::cuda::GpuMat d_descriptors;
		cv::cuda::GpuMat d_keypointsIr1;
		cv::cuda::GpuMat d_descriptorsIr1;
		cv::cuda::GpuMat d_keypointsIr2;
		cv::cuda::GpuMat d_descriptorsIr2;
		//std::vector<cv::KeyPoint> keypoints;
		std::vector<cv::KeyPoint> keypointsIr1;
		std::vector<cv::KeyPoint> keypointsIr2;
		cv::Mat descriptorsIr1;
		cv::Mat descriptorsIr2;
		std::vector<std::vector<cv::DMatch>> matches;

		double cx;
		double cy;
		double fx;
		double fy;
		cv::Mat intrinsic;
		cv::Mat distCoeffs;
		cv::Mat Rvec;
		cv::Mat t;

		Keyframe * currentKeyframe;
		bool keyframeExist;

		Gyro gyro;
		Accel accel;
		Pose pose;
	};

	// MultiCamera fixed
	Device device0;
	Device device1;
	std::string device0SN;
	std::string device1SN;

	// MultiCamera random
	std::vector<rs2::pipeline*> pipelines;
	std::vector<rs2::frameset> framesets;
	rs2::spatial_filter spatialFilter;

	rs2::align alignToColor = rs2::align(RS2_STREAM_COLOR);
	rs2::colorizer colorizer;
	
	int height;
	int width;
	int fps;
	
	Viewer * viewer;

	lup::Upsampling * upsampling;
	float maxDepth;

	// For Slam
	int minHessian;
	cv::cuda::SURF_CUDA surf;
	cv::Ptr<cv::cuda::ORB> orb;
	cv::Ptr< cv::cuda::DescriptorMatcher > matcher;
	FeatureDetectionMethod featMethod;

	// For Stereo Matching
	std::vector<cv::KeyPoint> stereoKeypointsIr1;
	std::vector<cv::KeyPoint> stereoKeypointsIr2;
	std::vector<cv::Point2f> stereoPointsIr1;
	std::vector<cv::Point2f> stereoPointsIr2;
	std::vector<float> stereoDistances;

	std::vector<Keyframe> keyframes;

	int relativeMatchingDefaultStereo(Device &device, Keyframe *keyframe, cv::Mat currentFrame);
	//int detectAndComputeOrb(Device &device);
	int detectAndComputeOrb(cv::Mat im, cv::cuda::GpuMat &d_im, std::vector<cv::KeyPoint> &keypoints, cv::cuda::GpuMat &descriptors);
	int solveRelativePose(Device& device, Keyframe *keyframe);
	int matchAndPose(Device& device);

	// Functions
	int initialize(Settings settings, FeatureDetectionMethod featMethod, std::string device0SN, std::string device1SN);

private:
	int initialize(int width, int height, int fps);
	int initialize(Settings settings);
	int setIntrinsics(Device &device, double cx, double cy, double fx, double fy);
	int initContainers(Device &device);

public:
	int recordAll();
	int playback(const char* serialNumber);
	int run(); // poseSolver thread
	int poseSolver(); // main loop for solving pose
	int poseSolverDefaultStereo();
	int poseSolverDefaultStereoMulti();
	int extractGyroAndAccel(Device &device);
	int extractColor(Device &device);
	int extractDepth(Device &device);
	int extractIr(Device &device);
	int extractTimeStamps();
	int upsampleDepth(Device &device);

	void updatePose();
	int getPose(); // fetcher of current pose

	// Utilities
	void visualizeImu(Device &device);
	void visualizePose();
	void updateViewerPose();
	void visualizeColor(Device &device);
	void visualizeDepth(Device &device);
	void visualizeFps(double fps);

	void visualizeKeypoints(cv::Mat im);
	void visualizeKeypoints(cv::Mat ir1, cv::Mat ir2);
	void visualizeRelativeKeypoints(Keyframe *keyframe, cv::Mat ir1, std::string windowNamePrefix);
	void visualizeStereoKeypoints(cv::Mat ir1, cv::Mat ir2);
	void visualizeRelativeKeypoints(Keyframe *keyframe, cv::Mat ir2);
	
	cv::Mat gyroDisp;
	cv::Mat accelDisp;

	//Tools
	std::string parseDecimal(double f);
	std::string parseDecimal(double f, int precision);
	void overlayMatrix(cv::Mat &im, cv::Mat R1, cv::Mat t);

	// Unused
	int solveKeypointsAndDescriptors(cv::Mat im);
	int solveStereoSurf(cv::Mat ir1, cv::Mat ir2);
	int solveStereoOrb(cv::Mat ir1, cv::Mat ir2);
	int solveRelativeSurf(Keyframe * keyframe);
	int solveRelativeOrb(Keyframe * keyframe);
	int detectAndComputeSurf(cv::Mat im, cv::cuda::GpuMat &d_im, std::vector<cv::KeyPoint> &keypoints, cv::cuda::GpuMat &descriptors);

	// Tests
	int testOrb();
	int testT265();
	int runTestViewerSimpleThread();
	int testViewerSimple();
	int getFrames();
	int testStream();
	int testImu();
	int getGyro(float *roll, float* pitch, float *yaw);

	int showAlignedDepth();
	int showDepth();
	static int getPose(float *x, float *y, float *z, float *roll, float *pitch, float *yaw);


};







// Trash codes
