#pragma once
#include <librealsense2\rs.hpp>
//#include <camerapose\CameraPose.h>
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

	class Pose {
	public:
		float x;
		float y;
		float z;
		float rx;
		float ry;
		float rz;
		float rw;
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

	class Device {
	public:
		Device() {};
		~Device() {};
		std::string id;
		bool isFound = false;
		std::string serialNo;

		rs2::pipeline * pipe;
		rs2::frameset frameset;
		rs2::config cfg;
		cv::Mat depth;
		cv::Mat depth32f;
		cv::Mat color;
		cv::Mat depthVis;
		cv::Mat infrared1;
		cv::Mat infrared2;
		cv::Mat infrared132f;
		cv::Mat infrared232f;
		float depthScale;

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
		cv::Mat mask;
		cv::cuda::GpuMat d_mask;

		double cx;
		double cy;
		double fx;
		double fy;
		cv::Mat intrinsic;
		cv::Mat distCoeffs;
		
		// Relative pose from keyframe
		cv::Mat Rvec;
		cv::Mat t;
		// Relative pose of IMU (from keyframe?)
		cv::Mat ImuRvec;
		cv::Mat ImuT;
		Quaternion ImuRotation;
		Vector3 ImuTranslation;
		Vector3 ImuVelocity;
		//bool imuKeyframeExist; //imu is global except yaw

		Keyframe * currentKeyframe;
		bool keyframeExist;

		Gyro gyro;
		Accel accel;
		Pose pose;

		rs2::frame_queue gyroQueue = rs2::frame_queue(1);
		rs2::frame_queue accelQueue = rs2::frame_queue(1);
		rs2::frame_queue depthQueue = rs2::frame_queue(1);
		rs2::frame_queue infrared1Queue = rs2::frame_queue(1);
		rs2::frame_queue infrared2Queue = rs2::frame_queue(1);
	};

	// MultiCamera fixed
	Device device0;
	Device device1;
	Device externalImu;
	std::string device0SN;
	std::string device1SN;

	// MultiCamera random
	std::vector<rs2::pipeline*> pipelines;
	std::vector<rs2::frameset> framesets;
	rs2::spatial_filter spatialFilter;
	rs2::context * ctx;

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

	int initialize(Settings settings, FeatureDetectionMethod featMethod, std::string device0SN, std::string device1SN);
	int initializeFromFile(const char* filename0, const char* filenameImu);
	int run(); // thread runner
	int runFromRecording();
	int singleThread();
	int fetchFrames(); // frameset fetcher thread
	int imuPoseSolver(); // imu thread
	int cameraPoseSolver(); // camera thread
	int poseRefinement(); // keyframe pose refinement thread

	int solveImuPose(Device& device);
	bool settleImu(Device& device);
	int matchAndPose(Device& device);
	int solveRelativePose(Device& device, Keyframe *keyframe);
	int createDepthThresholdMask(Device& device, float maxDepth);
	int detectAndComputeOrb(cv::Mat im, cv::cuda::GpuMat &d_im, std::vector<cv::KeyPoint> &keypoints, cv::cuda::GpuMat &descriptors);
	int detectAndComputeOrb(Device& device);
	int relativeMatchingDefaultStereo(Device &device, Keyframe *keyframe, cv::Mat currentFrame);
	
private:
	int initialize(int width, int height, int fps);
	int initialize(Settings settings);
	int setIntrinsics(Device &device, double cx, double cy, double fx, double fy);
	int initContainers(Device &device);
	bool isThisDevice(std::string serialNo, std::string queryNo);

public:
	bool processGyro(Device &device);
	bool processAccel(Device &device);
	bool processDepth(Device &device);
	bool processIr(Device &device);

	int upsampleDepth(Device &device);
	int extractColor(Device &device);
	int adjustGamma(Device &device);
	cv::Mat lookUpTable;

	/// Utilities
	void visualizeImu(Device &device);
	void visualizePose();
	void toEuler(Quaternion q, Vector3 &euler);
	// Convert Euler(in IMU coordinates) to Quaternion (in MADGWICK/WIKIPEDIA coordinates)
	void toQuaternion(Vector3 euler, Quaternion &q);
	void updateViewerCameraPose(Device &device);
	void updateViewerImuPose(Device &device);
	void visualizeColor(Device &device);
	void visualizeDepth(Device &device);
	void visualizeFps(double fps);
	void visualizeRelativeKeypoints(Keyframe *keyframe, cv::Mat ir1, std::string windowNamePrefix);
	
	cv::Mat gyroDisp;
	cv::Mat accelDisp;

	////Tools
	std::string parseDecimal(double f);
	std::string parseDecimal(double f, int precision);
	void overlayMatrix(const char * windowName, cv::Mat &im, cv::Mat R1, cv::Mat t);
	void overlayMatrixRot(const char* windowName, cv::Mat &im, Vector3 euler, Quaternion q);

	int saveImu(const char* filename0, const char* filenameImu, std::string outputFolder);
	int saveExternalImu(const char* filename0, const char* filenameImu, std::string outputFolder);
	int getSynchronization(const char* filename0, const char* filenameImu, std::string outputFolder);
	int saveAllDepthAndInfrared(const char* filename0, const char* filenameImu, std::string outputFolder);

	std::mutex mutex;

	/// Tests
	int testT265();
};