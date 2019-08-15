#pragma once

#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <device_launch_parameters.h>

class Stereo {
public:
	Stereo();
	Stereo(int BlockWidth, int BlockHeight, int StrideAlignemnt);
	~Stereo() {};

	int BlockWidth;
	int BlockHeight;
	int StrideAlignment;

	// Vector Fields
	cv::Mat translationVectorX;
	cv::Mat translationVectorY;
	cv::Mat calibrationVectorX;
	cv::Mat calibrationVectorY;
	float *d_tvxForward;
	float *d_tvyForward;
	float *d_tvxBackward;
	float *d_tvyBackward;
	float *d_tvx2; // vector where u,v points
	float *d_tvy2;
	float *d_cvx;
	float *d_cvy;
	float *d_i1calibrated;

	float *d_Iw;
	float *d_w; // Final stereo disparity
	float *d_ws;
	float *d_dw;
	float *d_dws;
	float *d_dwmed;
	float *d_dwmeds;
	float *d_pw1;
	float *d_pw2;
	float *d_pw1s;
	float *d_pw2s;
	std::vector<float*> pTvxForward;
	std::vector<float*> pTvyForward;
	std::vector<float*> pTvxBackward;
	std::vector<float*> pTvyBackward;
	cv::Mat disparity;
	cv::Mat depth;
	float *d_depth;

	float *d_wForward;
	float *d_wBackward;
	float *d_wFinal;
	float *d_depthFinal;
	float *d_occlusion;

	bool isReverse = false;
	bool isOcclusionChecked = false;
	bool isPlaneSweepOcclusionChecked = false;

	// Plane sweep
	float * ps_i1warp;
	float * ps_i1warps;
	float * ps_error;
	float * ps_depth;
	float * ps_disparity;
	int planeSweepMaxDisparity;
	int planeSweepWindow;
	float planeSweepMaxError;
	int planeSweepStride;
	cv::Mat planeSweepDepth;

	float * ps_depthFinal;
	float * ps_disparityForward;
	float * ps_disparityBackward;
	float * ps_disparityFinal;

	float baseline;
	float focal;

	// Stereo
	int loadVectorFields(cv::Mat translationVector, cv::Mat calibrationVector);
	int solveStereoForward();
	int solveStereoBackward(); // Can't call without calling forward
	int planeSweepForward();
	int planeSweepBackward();
	int copyStereoToHost(cv::Mat &w);
	int copyPlaneSweepToHost(cv::Mat &ps);
	int occlusionCheck(float threshold);
	int planeSweepOcclusionCheck();

	// Optical Flow
	int initializeFisheyeStereo(int width, int height, int channels, int inputType, int nLevels, float scale, float lambda,
		float theta, float tau, int nWarpIters, int nSolverIters);
	int initializeOpticalFlow(int width, int height, int channels, int inputType, int nLevels, float scale, float lambda,
		float theta, float tau, int nWarpIters, int nSolverIters);
	int copyOpticalFlowToHost(cv::Mat &u, cv::Mat &v);
	int solveOpticalFlow();
	int copyOpticalFlowVisToHost(cv::Mat &uvrgb);

	// Common
	int copyImagesToDevice(cv::Mat i0, cv::Mat i1);
	
	int width;
	int height;
	int stride;
	int inputType;
	float fScale;
	int nLevels;
	int inputChannels;
	int nSolverIters;
	int nWarpIters;

	float lambda;
	float theta;
	float tau;

	// Additional settings to limit compiling .cu's
	bool visualizeResults = false;
	float flowScale = 40.0f;

	cv::Mat im0pad, im1pad;
	cv::Mat uvrgb;
	cv::Mat upad, vpad;

	std::vector<float*> pI0;
	std::vector<float*> pI1;
	std::vector<int> pW;
	std::vector<int> pH;
	std::vector<int> pS;
	std::vector<int> pDataSize;
	int dataSize;
	int dataSize8uc3;
	int dataSize8u;
	int dataSize16u;
	int dataSize32f;
	int dataSize32fc3;

	float *d_i1warp;
	float *d_ix1warp;
	float *d_iy1warp;

	float *d_du;
	float *d_dv;
	float *d_dus;
	float *d_dvs;

	float *d_dumed;
	float *d_dvmed;
	float *d_dumeds;
	float *d_dvmeds;

	//dual TV
	float *d_pu1;
	float *d_pu2;
	float *d_pv1;
	float *d_pv2;
	//dual TV temps
	float *d_pu1s;
	float *d_pu2s;
	float *d_pv1s;
	float *d_pv2s;

	float *d_Ix;
	float *d_Iy;
	float *d_Iz;

	float *d_us;
	float *d_vs;

	//outputs
	float *d_u; //optical flow x
	float *d_v; //optical flow y
	float *d_uForward;
	float *d_vForward;

	//inputs
	// CV_8UC3
	uchar3 *d_i08uc3;
	uchar3 *d_i18uc3;
	//CV_8U
	uchar *d_i08u;
	uchar *d_i18u;

	// colored uv, for display only
	float3 *d_uvrgb;
	float3 *d_colorwheel;

	// Stereo Kernels
	void OcclusionCheck(float* wForward, float* wBackward, float threshold, 
		float *u, float *v, int w, int h, int s, float* wFinal);
	void ScalarMultiply(float *src, float scalar, int w, int h, int s);
	void ScalarMultiply(float *src, float scalar, int w, int h, int s, float *dst);
	void LimitRange(float *src, float upperLimit, int w, int h, int s, float *dst);
	void SetValue(float *image, float value, int w, int h, int s);
	void Clone(const float *src, int w, int h, int s, float *dst);
	void PlaneSweepCorrelation(float *i0, float *i1, float* disparity, int sweepDistance, int windowSize,
		int w, int h, int s,
		float *error);
	void FindWarpingVector(const float *u, const float *v, const float *tvx, const float *tvy, int w, int h, int s,
		float *tvx2, float *tvy2);
	void ComputeOpticalFlowVector(const float *dw, const float *tvx2, const float *tvy2,
		int w, int h, int s,
		float *du, float *dv);
	void ComputeDerivativesFisheye(float *I0, float *I1, float *vectorX, float *vectorY,
		int w, int h, int s, float *Iw, float *Iz);
	void MedianFilterDisparity(float *inputu,
		int w, int h, int s, float *outputu, int kernelsize);
	void SolveDataL1Stereo(const float *dwhat0,
		const float *pw1, const float *pw2,
		const float *Iw, const float *Iz,
		int w, int h, int s,
		float lambda, float theta,
		float *dwhat1);
	void SolveSmoothDualTVGlobalStereo(float *duhat,
		float *pw1, float *pw2,
		int w, int h, int s,
		float tau, float theta,
		float *pw1s, float*pw2s);
	void ConvertDisparityToDepth(float *disparity, float baseline, float focal, int w, int h, int s, float *depth);

	// Optical Flow Kernels
	void Cv8uToGray(uchar * d_iCv8u, float *d_iGray, int w, int h, int s);
	void Cv8uc3ToGray(uchar3 * d_iRgb, float *d_iGray, int w, int h, int s);
	void Downscale(const float *src, int width, int height, int stride,
		int newWidth, int newHeight, int newStride, float *out);
	void Downscale(const float *src, int width, int height, int stride,
		int newWidth, int newHeight, int newStride, float scale, float *out);
	void WarpImage(const float *src, int w, int h, int s,
		const float *u, const float *v, float *out);
	void ComputeDerivatives(float *I0, float *I1,
		int w, int h, int s, float *Ix, float *Iy, float *Iz);
	void Add(const float *op1, const float *op2, int count, float *sum);
	void MedianFilter3D(float *X, float *Y, float *Z,
		int w, int h, int s,
		float *X1, float *Y1, float *Z1,
		int kernelsize);
	void MedianFilter(float *inputu, float *inputv,
		int w, int h, int s,
		float *outputu, float*outputv,
		int kernelsize);
	void SolveDataL1(const float *duhat0, const float *dvhat0,
		const float *pu1, const float *pu2,
		const float *pv1, const float *pv2,
		const float *Ix, const float *Iy, const float *Iz,
		int w, int h, int s,
		float lambda, float theta,
		float *duhat1, float *dvhat1);
	void SolveSmoothDualTVGlobal(float *duhat, float *dvhat,
		float *pu1, float *pu2, float *pv1, float *pv2,
		int w, int h, int s,
		float tau, float theta,
		float *pu1s, float*pu2s,
		float *pv1s, float *pv2s);
	void Upscale(const float *src, int width, int height, int stride,
		int newWidth, int newHeight, int newStride, float scale, float *out);
	void FlowToHSV(float* u, float * v, int w, int h, int s, float3 * uRGB, float flowscale);


	// Utilities
	inline int iAlignUp(int n);
	int iDivUp(int n, int m);
	template<typename T> inline void Swap(T &a, T &ax);
	template<typename T> inline void Swap(T &a, T &ax, T &b, T &bx);
	template<typename T> inline void Swap(T &a, T &ax, T &b, T &bx, T &c, T &cx, T &d, T &dx);
	int computePyramidLevels(int width, int height, int minWidth, float scale);
	int initializeColorWheel();
};