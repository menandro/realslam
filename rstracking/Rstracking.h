#pragma once
#include <librealsense2\rs.hpp>
#include <camerapose\CameraPose.h>
#include <opencv2/opencv.hpp>
#include "lib_link.h"
#include <viewer\Viewer.h>
#include <filereader\Filereader.h>
#include <thread>

class Rstracking {
public:
	Rstracking() {};
	~Rstracking() {};

	rs2::pipeline * pipe;
	rs2::context * ctx;

	int width;
	int height;

	cv::Mat cameraLeft;
	cv::Mat cameraRight;
	cv::Mat intrinsicLeft;
	cv::Mat intrinsicRight;
	cv::Mat distortionLeft;
	cv::Mat distortionRight;

	int initialize();
	int testFisheye();
	int testFeatureDetect();
	int testFeatureDetectUndistort();
};
