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

class DirectAlignment {
public:
	DirectAlignment();
	~DirectAlignment() {};
	int BlockWidth, BlockHeight, StrideAlignment;

	int width;
	int height;
	int stride;
	float lambda;
	float theta;
	float tau;
	int nWarpIters;
	int nSolverIters;

	// Data sizes
	int dataSize8u;
	int dataSize8uc3;
	int dataSize32f;
	int dataSize32fc2;
	int dataSize32fc3;

	// Inputs and Outputs
	cv::Mat im0pad, im1pad;
	float *d_i0, *d_i1, *d_i1warp;
	uchar3 *d_i08uc3, *d_i18uc3;
	uchar *d_i08u, *d_i18u;

	float *d_Ix, *d_Iy, *d_Iz;
	float2 *d_u, *d_us, *d_umed; //optical flow
	float2 *d_du, *d_dumed;
	float2 *d_pu, *d_pus;
	float2 *d_pv, *d_pvs;
	cv::Mat maskPad;
	float *d_mask;
	cv::Mat u;
	float * d_grad;
	float gradThreshold = 0.15f;

	// Debugging
	cv::Mat uvrgb;
	float3 *d_uvrgb;

	int initialize(int width, int height, float lambda, float theta, float tau, int nSolverIters);
	int copyImagesToDevice(cv::Mat i0, cv::Mat i1);
	int copyMaskToDevice(cv::Mat mask);
	int solveDirectFlow();
	int solveDirectFlowNoPyr();
	int copyFlowToHost(cv::Mat &wCropped);
	int copyFlowColorToHost(cv::Mat &wCropped, float flowscale);

	// Kernels
	void WarpImage(const float *src, int w, int h, int s, const float2 *warpUV, float *out);
	void Cv8uToGray(uchar * d_iCv8u, float *d_iGray, int w, int h, int s);
	void Cv8uc3ToGray(uchar3 * d_iRgb, float *d_iGray, int w, int h, int s);
	void ThresholdingL1Masked(float* mask, float2* u, float2* u_, float* Ix, float* Iy, float* Iz,
		float lambda, float theta, int w, int h, int s);
	void SparsePriorL1(float* fisheyeMask, float* u, float* u_, float * usparse, float* Iu, float* Iz,
		float lambda, float l2lambda, float* lambdaMask, float theta,
		float* us, int w, int h, int s);
	void SimpleL2(float* mask, float* u, float* u_, float l2lambda, float* lambdaMask,
		float* us, int w, int h, int s);
	void SolveProblem1bMasked(float* mask, float2* u, float2 *pu, float2* pv, float theta,
		float2* umed, int w, int h, int s);
	void SolveProblem2Masked(float* mask, float2* u, float2 *pu, float2 *pv, float theta, float tau,
		float2* pus, float2* pvs, int w, int h, int s);
	void ComputeDerivatives(float *I0, float *I1,
		int w, int h, int s, float *Ix, float *Iy, float *Iz);
	void Gradient(float* input, int w, int h, int s, float* output);
	void FilterGradient(float* gradient, float2* u, float2* umed, float threshold, int w, int h, int s);
	void FlowToHSV(float2* uv, int w, int h, int s, float3 * uRGB, float flowscale);

	// UTILITIES
	int iAlignUp(int n);
	int iDivUp(int n, int m);
	template<typename T> void Swap(T &a, T &ax);
};