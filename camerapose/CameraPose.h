#ifndef CAMERA_POSE_H
#define CAMERA_POSE_H

#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <opencv2/tracking.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/cuda.hpp>
#include <opencv2/cudafeatures2d.hpp>

class CameraPose {
public:
	CameraPose();
	~CameraPose();

	int initialize(cv::Mat intrinsic);
	int initialize(cv::Mat intrinsic, int minHessian);
	int initialize(double focal, cv::Point2d pp);
	int initialize(double focal, cv::Point2d pp, int minHessian);
	int initialize(double focalx, double focaly, cv::Point2d pp, int minHessian);

	int solvePose_8uc3(cv::Mat im1, cv::Mat im2, cv::Mat &R, cv::Mat &t);
	int solvePose_8uc1(cv::Mat im1, cv::Mat im2, cv::Mat &R, cv::Mat &t);

	int drawKeypoints();
	void overlay_matrix(cv::Mat &im, cv::Mat R1, cv::Mat t);
	std::string parse_decimal(double f);

	cv::Mat rotation; // 3x3 matrix
	cv::Mat translation;
	cv::Mat R;
	cv::Mat t;

	cv::Mat intrinsic; //3x3 matrix
	cv::Mat K;

	double focal;
	cv::Point2d pp;

	cv::cuda::GpuMat d_im1;
	cv::cuda::GpuMat d_im2;
	cv::cuda::GpuMat d_keypoints_im1, d_keypoints_im2; // keypoints
	cv::cuda::GpuMat d_descriptors_im1, d_descriptors_im2;

	cv::Mat im1, im2;
	cv::Mat im1_rgb, im2_rgb;
	cv::Mat im1_draw, im2_draw;
	std::vector< cv::KeyPoint > keypoints_im1, keypoints_im2;
	cv::Mat pose;

	int minHessian = 1000;
	cv::cuda::SURF_CUDA surf;

	cv::Ptr< cv::cuda::DescriptorMatcher > matcher;
	std::vector< std::vector< cv::DMatch> > matches;

	std::vector<cv::Point2f> matchedpoints_im1;
	std::vector<cv::Point2f> matchedpoints_im2;
	std::vector<cv::Point2f> filtered_matchedpoints_im1;
	std::vector<cv::Point2f> filtered_matchedpoints_im2;
	std::vector< cv::KeyPoint > matched_keypoints_im1;
	std::vector< cv::KeyPoint > matched_keypoints_im2;
	std::vector< float > matched_distance;
	std::vector< cv::KeyPoint > filtered_keypoints_im1;
	std::vector< cv::KeyPoint > filtered_keypoints_im2;

	cv::Mat E;
	cv::Mat keypoint_mask;
};

#endif
