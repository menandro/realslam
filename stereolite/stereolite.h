#pragma once
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <memory.h>
#include <math.h>

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <device_launch_parameters.h>

#include "lib_link.h"

class StereoLite {
public:
	StereoLite();
	StereoLite(int blockWidth, int blockHeight, int strideAlignment);
	~StereoLite() {};
	int BlockWidth, BlockHeight, StrideAlignment;

	int width;
	int height;
	int stride;
	float lambda;
	float theta;
	float tau;
	float fScale;
	int nLevels;
	int nWarpIters;
	int nSolverIters;
	float limitRange = 1.0f;
	float baseline;
	float focal;

	// Data sizes
	int dataSize8u;
	int dataSize8uc3;
	int dataSize32f;
	int dataSize32fc2;
	int dataSize32fc3;
	int dataSize32fc4;

	// Inputs and Outputs
	cv::Mat im0pad, im1pad;
	float *d_i0, *d_i1, *d_i1warp;
	uchar3 *d_i08uc3, *d_i18uc3;
	uchar *d_i08u, *d_i18u;

	float *d_Iu, *d_Iz;
	// Output Disparity
	float* d_u, *d_du, *d_us;
	// Output Depth
	float* d_depth;
	cv::Mat depth;
	// Warping Variables
	float2 *d_warpUV, *d_warpUVs, *d_dwarpUV;
	cv::Mat warpUV, warpUVrgb;
	float3 *d_uvrgb;

	// TVL1 Process variables
	float2 *d_p, *d_ps;
	float *d_u_, *d_u_s, *d_u_last;

	// Pyramid
	std::vector<float*> pI0;
	std::vector<float*> pI1;
	std::vector<int> pW;
	std::vector<int> pH;
	std::vector<int> pS;
	std::vector<int> pDataSize;
	cv::Mat fisheyeMaskPad;
	float* d_fisheyeMask;
	std::vector<float*> pFisheyeMask;

	// Vector Fields
	cv::Mat translationVector;
	cv::Mat calibrationVector;
	float2 *d_tvForward;
	float2 *d_tvBackward;
	float2 *d_tv2;
	float2 *d_cv;
	float *d_i1calibrated;
	std::vector<float2*> pTvForward;
	std::vector<float2*> pTvBackward;

	// 3D
	float3 *d_X;

	// Debug
	float *debug_depth;

	int initialize(int width, int height, float lambda, float theta, float tau,
		int nLevels, float fScale, int nWarpIters, int nSolverIters);
	int copyMaskToDevice(cv::Mat mask);
	int copyImagesToDevice(cv::Mat i0, cv::Mat i1);
	int loadVectorFields(cv::Mat translationVector, cv::Mat calibrationVector);
	int solveStereoForwardMasked();
	int copyStereoToHost(cv::Mat &wCropped);
	int copyPlanesweepForwardToHost(cv::Mat &wCropped);
	int copyPlanesweepBackwardToHost(cv::Mat &wCropped);

	// Planesweep
	int planeSweepForward();
	int planeSweepBackward();
	int planeSweep();
	float * ps_i1warp;
	float * ps_i1warps;
	float * ps_error;
	float * ps_depth;
	float * ps_disparityForward;
	float * ps_disparityBackward;
	float * ps_disparityFinal;
	int planeSweepMaxDisparity;
	int planeSweepWindow;
	float planeSweepMaxError;
	int planeSweepStride;
	cv::Mat planeSweepDepth;

	// UTILITIES
	int iAlignUp(int n);
	int iDivUp(int n, int m);
	template<typename T> void Swap(T &a, T &ax);
	template<typename T> void Copy(T &dst, T &src);

	// Kernels
	// Planesweep
	void PlaneSweepCorrelation(float *i0, float *i1, float* disparity, int sweepDistance, int windowSize,
		int w, int h, int s, float *error);
	void SetValue(float *image, float value, int w, int h, int s);

	// TVL1
	void ThresholdingL1Masked(float* mask, float* u, float* u_, float* Iu, float* Iz, float lambda, float theta,
		float* us, int w, int h, int s);
	void SolveProblem1bMasked(float* mask, float* u, float2 *p, float theta,
		float* us, int w, int h, int s);
	void SolveProblem2Masked(float* mask, float* u, float2 *p, float theta, float tau,
		float2* ps, int w, int h, int s);

	void ConvertDisparityToDepth(float *disparity, float baseline, float focal, int w, int h, int s, float *depth);
	void FlowToHSV(float2* uv, int w, int h, int s, float3 * uRGB, float flowscale);
	void MedianFilter(float *inputu, float *inputv, int w, int h, int s,
		float *outputu, float*outputv, int kernelsize);
	void MedianFilter3D(float *X, float *Y, float *Z, int w, int h, int s,
		float *X1, float *Y1, float *Z1, int kernelsize);
	void MedianFilterDisparity(float *input, int w, int h, int s,
		float *outputu, int kernelsize);
	void Cv8uToGray(uchar * d_iCv8u, float *d_iGray, int w, int h, int s);
	void Cv8uc3ToGray(uchar3 * d_iRgb, float *d_iGray, int w, int h, int s);
	void Upscale(const float *src, int width, int height, int stride,
		int newWidth, int newHeight, int newStride, float scale, float *out);
	void Upscale(const float2 *src, int width, int height, int stride,
		int newWidth, int newHeight, int newStride, float scale, float2 *out);
	void LimitRange(float *src, float upperLimit, int w, int h, int s, float *dst);
	void ComputeDerivatives(float *I0, float *I1,
		int w, int h, int s,
		float *Ix, float *Iy, float *Iz);
	void ComputeDerivativesFisheye(float *I0, float *I1, float2 *vector,
		int w, int h, int s, float *Iw, float *Iz);
	void Clone(float2* dst, int w, int h, int s, float2* src);
	void Clone(float* dst, int w, int h, int s, float* src);
	void Downscale(const float *src, int width, int height, int stride,
		int newWidth, int newHeight, int newStride, float *out);
	void DownscaleNearestNeighbor(const float *src, int width, int height, int stride,
		int newWidth, int newHeight, int newStride, float *out);
	void Downscale(const float2 *src, int width, int height, int stride,
		int newWidth, int newHeight, int newStride, float2 *out);
	void Downscale(const float *src, int width, int height, int stride,
		int newWidth, int newHeight, int newStride, float scale, float *out);
	void Downscale(const float2 *src, int width, int height, int stride,
		int newWidth, int newHeight, int newStride, float scale, float2 *out);
	void Subtract(float *minuend, float* subtrahend, int w, int h, int s, float* difference);
	void Add(float2 *src1, float2* src2, int w, int h, int s, float2* dst);
	void Add(float *src1, float* src2, int w, int h, int s, float* dst);
	void ScalarMultiply(float2 *src, float scalar, int w, int h, int s, float2* dst);
	void ScalarMultiply(float *src, float scalar, int w, int h, int s, float* dst);
	void ScalarMultiply(float *src, float scalar, int w, int h, int s);
	void WarpImage(const float *src, int w, int h, int s,
		const float2 *warpUV, float *out);
	void FindWarpingVector(const float2 *warpUV, const float *tvx, const float *tvy,
		int w, int h, int s, float2 *tv2);
	void FindWarpingVector(const float2 *warpUV, const float2 *tv,
		int w, int h, int s, float2 *tv2);
	void ComputeOpticalFlowVector(const float *u, const float2 *tv2,
		int w, int h, int s, float2 *warpUV);
};

