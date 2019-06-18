#pragma once
#include <opencv2/opencv.hpp>

#ifdef _DEBUG
#define LIB_EXT "d.lib"
#else
#define LIB_EXT ".lib"
#endif
#define CV_LIB_PATH "D:/dev/lib64/"
#define LIB_PATH "D:/dev/lib64/"
#define CV_VER_NUM  CVAUX_STR(CV_MAJOR_VERSION) CVAUX_STR(CV_MINOR_VERSION) CVAUX_STR(CV_SUBMINOR_VERSION)


#pragma comment(lib, "opengl32" LIB_EXT)
#pragma comment(lib, LIB_PATH "glfw3" LIB_EXT)
#pragma comment(lib, CV_LIB_PATH "opencv_core" CV_VER_NUM LIB_EXT)
#pragma comment(lib, CV_LIB_PATH "opencv_highgui" CV_VER_NUM LIB_EXT)
#pragma comment(lib, CV_LIB_PATH "opencv_videoio" CV_VER_NUM LIB_EXT)
//#pragma comment(lib, CV_LIB_PATH "opencv_viz" CV_VER_NUM LIB_EXT)
#pragma comment(lib, CV_LIB_PATH "opencv_cudafeatures2d" CV_VER_NUM LIB_EXT)
#pragma comment(lib, CV_LIB_PATH "opencv_cudaimgproc" CV_VER_NUM LIB_EXT)
#pragma comment(lib, CV_LIB_PATH "opencv_imgproc" CV_VER_NUM LIB_EXT)
#pragma comment(lib, CV_LIB_PATH "opencv_calib3d" CV_VER_NUM LIB_EXT)
#pragma comment(lib, CV_LIB_PATH "opencv_xfeatures2d" CV_VER_NUM LIB_EXT)
#pragma comment(lib, CV_LIB_PATH "opencv_optflow" CV_VER_NUM LIB_EXT)
#pragma comment(lib, CV_LIB_PATH "opencv_imgcodecs" CV_VER_NUM LIB_EXT)
#pragma comment(lib, CV_LIB_PATH "opencv_features2d" CV_VER_NUM LIB_EXT)
#pragma comment(lib, CV_LIB_PATH "opencv_tracking" CV_VER_NUM LIB_EXT)
#pragma comment(lib, CV_LIB_PATH "opencv_flann" CV_VER_NUM LIB_EXT)
