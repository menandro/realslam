#pragma once

#include <opencv2/opencv.hpp>

#if _WIN64
#define LIB_PATH "D:/dev/lib64/"
#else
#define LIB_PATH "D:/dev/staticlib32/"
#endif

#ifdef _DEBUG
#define LIB_EXT "d.lib"
#else
#define LIB_EXT ".lib"
#endif

//realsense
#pragma comment(lib, "C:/Program Files (x86)/Intel RealSense SDK 2.0/lib/x64/realsense2.lib")

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
////#pragma comment(lib, LIB_PATH "zlib.lib")
////#pragma comment(lib, LIB_PATH "comctl32.lib")
//
//// remove for hololens
////kernel32.lib; user32.lib; gdi32.lib; winspool.lib; comdlg32.lib; 
////advapi32.lib; shell32.lib; ole32.lib; oleaut32.lib; uuid.lib; odbc32.lib; odbccp32.lib
//#pragma comment(lib, "kernel32.lib")
//#pragma comment(lib, "user32.lib")
//#pragma comment(lib, "shell32.lib")
//#pragma comment(lib, "gdi32.lib")
//#pragma comment(lib, "winspool.lib")
//#pragma comment(lib, "comdlg32.lib")
//#pragma comment(lib, "advapi32.lib")
//#pragma comment(lib, "ole32.lib")
//#pragma comment(lib, "oleaut32.lib")
//#pragma comment(lib, "uuid.lib")
//#pragma comment(lib, "odbc32.lib")
//#pragma comment(lib, "odbccp32.lib")