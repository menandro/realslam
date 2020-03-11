#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d/cuda.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/optflow.hpp>

#include "lib_link.h"

#include <viewer\Viewer.h>
#include <filereader\Filereader.h>
#include <stereotgv/stereotgv.h>
#include <opticalflow/flow.h>
#include <RsT265/T265.h>
#include <imu/imu.h>

#include <thread>

#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>

class Keyframe {
public:
	Keyframe() {};
	~Keyframe() {};
	cv::Mat im;
	cv::Mat X;
	std::vector<cv::Point3f> objectPoints;

	// Global Pose
	cv::Mat R; // Rodrigues Rotation
	cv::Mat t;

	// Poses and Displays is always WRT to Keyframe
	cv::Mat Rrel; // Rodrigues Rotation
	cv::Mat trel;
};

class FisheyeSlam {
public:
	FisheyeSlam();
	~FisheyeSlam() {};

	// Inputs
	int width;
	int height;
	cv::Mat fisheye1;
	cv::Mat fisheye2;
	cv::Mat depth;
	cv::Mat point3d;
	cv::Mat fisheyeMask8uc1; //8u
	cv::Mat fisheyeMask32fc1;
	Quaternion imuRotation;

	cv::Mat intrinsic;
	cv::Mat distortionCoeffs;

	// Pose Viewer
	Viewer* viewer;
	void updateViewerImuPose();
	void updateViewerCameraPose(cv::Mat Rvec, cv::Mat t);

	// Point Cloud Viewer
	Viewer* pointcloudViewer;
	std::vector<float> pcVertexArray; //3-vertex, 3-normal, 2-texture uv
	std::vector<unsigned int> pcIndexArray; // 3-index (triangle);
	void pointcloudToArray(cv::Mat pc, std::vector<float>& vertexArray, std::vector<unsigned int>& indexArray);

	int initialize(int width, int height, cv::Mat intrinsic, cv::Mat distCoeffs, cv::Mat fisheyeMask8uc1);
	

	// Optical Flow
	Flow* flow;
	int flowWidth;
	int flowHeight;
	//float flowScaling;
	int initOpticalFlow();
	int solveOpticalFlow(cv::Mat im1, cv::Mat im2);
	cv::Mat flowTemp, flowTempMasked, flowTempRgb;

	// SLAM
	std::vector<Keyframe> keyframes;
	int currKFIndex;
	bool keyframeExist = false;
	//cv::Mat currKfImage;
	//cv::Mat currKfPoint3d;
	//std::queue<cv::Mat> images;

	cv::Mat currImage;
	cv::Mat prevImage; // For optical flow
	cv::Mat currX;
	Quaternion currImuRotation;
	std::vector<cv::Point3f> currObjectPoints;
	std::vector<cv::Point2f> currMatchedPoints;

	std::atomic<bool> isImageUpdated = false;
	std::atomic<bool> isImuUpdated = false;
	int run();
	int updateImageSlam(cv::Mat im, cv::Mat point3d);
	int updateImu(Quaternion imuRotation);
	int tracking();
	int mapping();
	int visualizePose();

	// Utilities
	void overlayMatrix(const char* windowName, cv::Mat& im, cv::Mat R1, cv::Mat t);
	std::string parseDecimal(double f);
	std::string parseDecimal(double f, int precision);
};