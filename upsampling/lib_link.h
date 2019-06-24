#pragma once

#include <opencv2/opencv.hpp>

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

#define CUDA_LIB_PATH "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.0/lib/x64/"
#pragma comment(lib, CUDA_LIB_PATH "cudart.lib")


#define CV_VER_NUM CVAUX_STR(CV_MAJOR_VERSION) CVAUX_STR(CV_MINOR_VERSION) CVAUX_STR(CV_SUBMINOR_VERSION)
#pragma comment(lib, CV_LIB_PATH "opencv_core" CV_VER_NUM LIB_EXT)
#pragma comment(lib, CV_LIB_PATH "opencv_highgui" CV_VER_NUM LIB_EXT)
#pragma comment(lib, CV_LIB_PATH "opencv_videoio" CV_VER_NUM LIB_EXT)
//#pragma comment(lib, CV_LIB_PATH "opencv_viz" CV_VER_NUM LIB_EXT)
//#pragma comment(lib, CV_LIB_PATH "opencv_cudafeatures2d" CV_VER_NUM LIB_EXT)
//#pragma comment(lib, CV_LIB_PATH "opencv_cudaimgproc" CV_VER_NUM LIB_EXT)
//#pragma comment(lib, CV_LIB_PATH "opencv_imgproc" CV_VER_NUM LIB_EXT)
//#pragma comment(lib, CV_LIB_PATH "opencv_calib3d" CV_VER_NUM LIB_EXT)
//#pragma comment(lib, CV_LIB_PATH "opencv_xfeatures2d" CV_VER_NUM LIB_EXT)
//#pragma comment(lib, CV_LIB_PATH "opencv_optflow" CV_VER_NUM LIB_EXT)
//#pragma comment(lib, CV_LIB_PATH "opencv_imgcodecs" CV_VER_NUM LIB_EXT)
//#pragma comment(lib, CV_LIB_PATH "opencv_features2d" CV_VER_NUM LIB_EXT)
//#pragma comment(lib, CV_LIB_PATH "opencv_tracking" CV_VER_NUM LIB_EXT)
//#pragma comment(lib, CV_LIB_PATH "opencv_ximgproc" CV_VER_NUM LIB_EXT)

//#pragma comment(lib, LIB_PATH "IlmImf.lib")
//#pragma comment(lib, LIB_PATH "ippicvmt.lib")
//#pragma comment(lib, LIB_PATH "ippiw.lib")
//#pragma comment(lib, LIB_PATH "ittnotify.lib")
//#pragma comment(lib, LIB_PATH "libjasper.lib")
//#pragma comment(lib, LIB_PATH "libjpeg-turbo.lib")
//#pragma comment(lib, LIB_PATH "libpng.lib")
//#pragma comment(lib, LIB_PATH "libprotobuf.lib")
//#pragma comment(lib, LIB_PATH "libtiff.lib")
//#pragma comment(lib, LIB_PATH "libwebp.lib")
//#pragma comment(lib, LIB_PATH "zlib.lib")


