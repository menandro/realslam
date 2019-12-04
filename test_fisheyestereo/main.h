#pragma once

#include <opencv2/opencv.hpp>
#include <stereotgv/stereotgv.h>
#include <stereolite/stereolite.h>
#include <time.h>

#if _WIN64
#define LIB_PATH "D:/dev/lib64/"
#define CV_LIB_PATH "D:/dev/lib64/"
#else
#define LIB_PATH "D:/dev/staticlib32/"
#endif

#ifdef _DEBUG
#define LIB_EXT "d.lib"
#else
#define LIB_EXT ".lib"
#endif

#define CUDA_LIB_PATH "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.1/lib/x64/"
#pragma comment(lib, CUDA_LIB_PATH "cudart.lib")

#define CV_VER_NUM CVAUX_STR(CV_MAJOR_VERSION) CVAUX_STR(CV_MINOR_VERSION) CVAUX_STR(CV_SUBMINOR_VERSION)
#pragma comment(lib, LIB_PATH "opencv_core" CV_VER_NUM LIB_EXT)
#pragma comment(lib, LIB_PATH "opencv_highgui" CV_VER_NUM LIB_EXT)
#pragma comment(lib, LIB_PATH "opencv_videoio" CV_VER_NUM LIB_EXT)
#pragma comment(lib, LIB_PATH "opencv_imgproc" CV_VER_NUM LIB_EXT)
#pragma comment(lib, LIB_PATH "opencv_calib3d" CV_VER_NUM LIB_EXT)
#pragma comment(lib, LIB_PATH "opencv_xfeatures2d" CV_VER_NUM LIB_EXT)
#pragma comment(lib, LIB_PATH "opencv_optflow" CV_VER_NUM LIB_EXT)
#pragma comment(lib, LIB_PATH "opencv_imgcodecs" CV_VER_NUM LIB_EXT)
#pragma comment(lib, LIB_PATH "opencv_features2d" CV_VER_NUM LIB_EXT)
#pragma comment(lib, LIB_PATH "opencv_tracking" CV_VER_NUM LIB_EXT)
#pragma comment(lib, LIB_PATH "opencv_flann" CV_VER_NUM LIB_EXT)
#pragma comment(lib, LIB_PATH "opencv_cudafeatures2d" CV_VER_NUM LIB_EXT)
#pragma comment(lib, LIB_PATH "opencv_cudaimgproc" CV_VER_NUM LIB_EXT)
#pragma comment(lib, LIB_PATH "opencv_video" CV_VER_NUM LIB_EXT)

// Revision
int test_BlenderPerspectiveDataSequence();
int test_SolveTrajectoryPerWarpingBlenderDataAll();
int test_SolveTrajectoryPerWarpingBlenderData();
int test_SolveTrajectoryPerWarpingFaro();
int test_BlenderDataAllPlanesweep();
int test_FaroDataAllPlanesweep();
int test_FaroDataAll();
int test_ImageSequencePlanesweep(std::string mainfolder, int startFrame, int endFrame);
int test_ImageSequence(std::string mainfolder, int startFrame, int endFrame);

int test_VehicleSegmentationSequence();
int test_VehicleSegmentation();
int test_PlaneSweepWithTvl1();

int test_PlaneSweep();
int test_StereoLiteTwoFrames(int nLevel, float fScale, int nWarpIters, int nSolverIters);
int test_TwoFrames(int nLevel, float fScale, int nWarpIters, int nSolverIters);
int test_LimitingRangeOne();
int test_BlenderDataSequence();

int test_ImageSequenceLite();
int test_Timing(int warpIteration);
int test_LimitingRange();
int test_FaroData();
int test_BlenderData();
int test_IcraAddedAccuratePixels();
void showDepthJet(std::string windowName, cv::Mat image, float maxDepth, bool shouldWait);
void showDepthJetExponential(std::string windowName, cv::Mat image, float maxDepth, float curve, bool shouldWait);
void saveDepthJetExponential(std::string fileName, cv::Mat image, float maxDepth, float curve);
void saveDepthJet(std::string fileName, cv::Mat image, float maxDepth);
int test_TwoImagesRealsense();