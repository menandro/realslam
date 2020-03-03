#include <opencv2/opencv.hpp>
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

#define CUDA_LIB_PATH "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/lib/x64/"
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


int main() {
	// THETA S
	//cv::VideoCapture webcam1;

	//webcam1.open(0, cv::CAP_DSHOW);

	///*webcam1.set(cv::CAP_PROP_FPS, 15);
	//webcam1.set(cv::CAP_PROP_FRAME_HEIGHT, 1440);
	//webcam1.set(cv::CAP_PROP_FRAME_WIDTH, 1440);*/

	//for (;;)
	//{
	//	cv::Mat frame1, frame2;
	//	webcam1 >> frame1;
	//	cv::imshow("frame1", frame1);
	//	//std::cout << frame1.cols << " " << frame1.rows << std::endl;
	//	if (cv::waitKey(10) == 27) break; // stop capturing by pressing ESC 
	//}

	//webcam1.release();

	// PIXPRO
	cv::VideoCapture webcam1;
	cv::VideoCapture webcam2;

	webcam1.open(0, cv::CAP_DSHOW);
	webcam2.open(1, cv::CAP_DSHOW);

	webcam1.set(cv::CAP_PROP_FPS, 15);
	webcam1.set(cv::CAP_PROP_FRAME_HEIGHT, 1440);
	webcam1.set(cv::CAP_PROP_FRAME_WIDTH, 1440);
	webcam2.set(cv::CAP_PROP_FPS, 15);
	webcam2.set(cv::CAP_PROP_FRAME_HEIGHT, 1440);
	webcam2.set(cv::CAP_PROP_FRAME_WIDTH, 1440);

	for (;;)
	{
		cv::Mat frame1, frame2;
		webcam1 >> frame1;
		webcam2 >> frame2;
		cv::imshow("frame1", frame1);
		cv::imshow("frame2", frame2);
		//std::cout << frame1.cols << " " << frame1.rows << std::endl;
		if (cv::waitKey(10) == 27) break; // stop capturing by pressing ESC 
	}

	webcam1.release();
	webcam2.release();

	return 0;
}