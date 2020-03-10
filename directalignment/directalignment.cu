#include "directalignment.h"

DirectAlignment::DirectAlignment() {
	this->BlockHeight = 1;
	this->BlockWidth = 32;
	this->StrideAlignment = 32;
}

int DirectAlignment::initialize(int width, int height, float lambda, float theta, float tau,
	int nSolverIters) {

	this->width = width;
	this->height = height;
	this->stride = this->iAlignUp(width);
	this->lambda = lambda;
	this->theta = theta;
	this->tau = tau;
	this->nSolverIters = nSolverIters;

	this->height = height;
	this->width = width;
	this->stride = iAlignUp(width);

	dataSize8u = stride * height * sizeof(uchar);
	dataSize8uc3 = stride * height * sizeof(uchar3);
	dataSize32f = stride * height * sizeof(float);
	dataSize32fc2 = stride * height * sizeof(float2);
	dataSize32fc3 = stride * height * sizeof(float3);

	// Inputs and Outputs
	checkCudaErrors(cudaMalloc(&d_i0, dataSize32f));
	checkCudaErrors(cudaMalloc(&d_i1, dataSize32f));
	checkCudaErrors(cudaMalloc(&d_i1warp, dataSize32f));
	checkCudaErrors(cudaMalloc(&d_i08u, dataSize8u));
	checkCudaErrors(cudaMalloc(&d_i18u, dataSize8u));
	checkCudaErrors(cudaMalloc(&d_i08uc3, dataSize8uc3));
	checkCudaErrors(cudaMalloc(&d_i18uc3, dataSize8uc3));
	checkCudaErrors(cudaMalloc(&d_Ix, dataSize32f));
	checkCudaErrors(cudaMalloc(&d_Iy, dataSize32f));
	checkCudaErrors(cudaMalloc(&d_Iz, dataSize32f));
	checkCudaErrors(cudaMalloc(&d_mask, dataSize32f));
	checkCudaErrors(cudaMalloc(&d_grad, dataSize32f));

	// Output Optical Flow
	checkCudaErrors(cudaMalloc(&d_u, dataSize32fc2));
	checkCudaErrors(cudaMalloc(&d_us, dataSize32fc2));
	checkCudaErrors(cudaMalloc(&d_umed, dataSize32fc2));

	checkCudaErrors(cudaMalloc(&d_du, dataSize32fc2));
	checkCudaErrors(cudaMalloc(&d_dumed, dataSize32fc2));

	// Process variables
	checkCudaErrors(cudaMalloc(&d_pu, dataSize32fc2));
	checkCudaErrors(cudaMalloc(&d_pus, dataSize32fc2));
	checkCudaErrors(cudaMalloc(&d_pv, dataSize32fc2));
	checkCudaErrors(cudaMalloc(&d_pvs, dataSize32fc2));

	// Debugging
	checkCudaErrors(cudaMalloc(&d_uvrgb, dataSize32fc3));
	uvrgb = cv::Mat(height, stride, CV_32FC3);
	u = cv::Mat(height, stride, CV_32FC2);

	return 0;
}

int DirectAlignment::copyImagesToDevice(cv::Mat i0, cv::Mat i1) {
	// Padding
	cv::copyMakeBorder(i0, im0pad, 0, 0, 0, stride - width, cv::BORDER_CONSTANT, 0);
	cv::copyMakeBorder(i1, im1pad, 0, 0, 0, stride - width, cv::BORDER_CONSTANT, 0);

	if (i0.type() == CV_8U) {
		checkCudaErrors(cudaMemcpy(d_i08u, (uchar *)im0pad.ptr(), dataSize8u, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_i18u, (uchar *)im1pad.ptr(), dataSize8u, cudaMemcpyHostToDevice));
		// Convert to 32F
		Cv8uToGray(d_i08u, d_i0, width, height, stride);
		Cv8uToGray(d_i18u, d_i1, width, height, stride);
	}
	else if (i0.type() == CV_32F) {
		checkCudaErrors(cudaMemcpy(d_i0, (float *)im0pad.ptr(), dataSize32f, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_i1, (float *)im1pad.ptr(), dataSize32f, cudaMemcpyHostToDevice));
	}
	else if (i0.type() == CV_8UC3) {
		checkCudaErrors(cudaMemcpy(d_i08uc3, (uchar3 *)im0pad.ptr(), dataSize8uc3, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_i18uc3, (uchar3 *)im1pad.ptr(), dataSize8uc3, cudaMemcpyHostToDevice));
		// Convert to 32F
		Cv8uc3ToGray(d_i08uc3, d_i0, width, height, stride);
		Cv8uc3ToGray(d_i18uc3, d_i1, width, height, stride);
	}
	return 0;
}

int DirectAlignment::copyMaskToDevice(cv::Mat mask) {
	cv::copyMakeBorder(mask, maskPad, 0, 0, 0, stride - width, cv::BORDER_CONSTANT, 0);
	checkCudaErrors(cudaMemcpy(d_mask, (float *)maskPad.ptr(), dataSize32f, cudaMemcpyHostToDevice));

	return 0;
}

int DirectAlignment::solveDirectFlow() {
	checkCudaErrors(cudaMemset(d_u, 0, dataSize32fc2));
	checkCudaErrors(cudaMemset(d_umed, 0, dataSize32fc2));
	checkCudaErrors(cudaMemset(d_pu, 0, dataSize32fc2));
	checkCudaErrors(cudaMemset(d_pv, 0, dataSize32fc2));
	checkCudaErrors(cudaMemset(d_pus, 0, dataSize32fc2));
	checkCudaErrors(cudaMemset(d_pvs, 0, dataSize32fc2));

	ComputeDerivatives(d_i0, d_i1, width, height, stride, d_Ix, d_Iy, d_Iz);
	Gradient(d_i0, width, height, stride, d_grad);
	// Solve optical flow
	for (int iter = 0; iter < nSolverIters; iter++) {
		// Solve Problem1A
		ThresholdingL1Masked(d_mask, d_u, d_umed, d_Ix, d_Iy, d_Iz, lambda, theta, width, height, stride);
		//Swap(d_u, d_us);

		// Solve Problem1B
		SolveProblem1bMasked(d_mask, d_u, d_pu, d_pv, theta, d_umed, width, height, stride);

		// Solve Problem2
		SolveProblem2Masked(d_mask, d_umed, d_pu, d_pv, theta, tau, d_pus, d_pvs, width, height, stride);
		Swap(d_pu, d_pus);
		Swap(d_pv, d_pvs);

		// Filter edges
		//FilterGradient(d_grad, d_u, d_umed, gradThreshold, width, height, stride);
	}
	return 0;
}

int DirectAlignment::solveDirectFlowNoPyr() {
	checkCudaErrors(cudaMemset(d_u, 0, dataSize32fc2));
	checkCudaErrors(cudaMemset(d_umed, 0, dataSize32fc2));
	checkCudaErrors(cudaMemset(d_pu, 0, dataSize32fc2));
	checkCudaErrors(cudaMemset(d_pv, 0, dataSize32fc2));
	checkCudaErrors(cudaMemset(d_pus, 0, dataSize32fc2));
	checkCudaErrors(cudaMemset(d_pvs, 0, dataSize32fc2));

	ComputeDerivatives(d_i0, d_i1, width, height, stride, d_Ix, d_Iy, d_Iz);
	Gradient(d_i0, width, height, stride, d_grad);
	// Solve optical flow
	for (int iter = 0; iter < nSolverIters; iter++) {
		// Solve Problem1A
		ThresholdingL1Masked(d_mask, d_u, d_umed, d_Ix, d_Iy, d_Iz, lambda, theta, width, height, stride);
		//Swap(d_u, d_us);

		// Solve Problem1B
		SolveProblem1bMasked(d_mask, d_u, d_pu, d_pv, theta, d_umed, width, height, stride);

		// Solve Problem2
		SolveProblem2Masked(d_mask, d_umed, d_pu, d_pv, theta, tau, d_pus, d_pvs, width, height, stride);
		Swap(d_pu, d_pus);
		Swap(d_pv, d_pvs);

		// Filter edges
		//FilterGradient(d_grad, d_u, d_umed, gradThreshold, width, height, stride);
	}
	return 0;
}

int DirectAlignment::copyFlowToHost(cv::Mat &wCropped) {
	// Remove Padding
	//checkCudaErrors(cudaMemcpy((float *)depth.ptr(), d_w, stride * height * sizeof(float), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy((float2 *)u.ptr(), d_umed, dataSize32fc2, cudaMemcpyDeviceToHost));
	cv::Rect roi(0, 0, width, height); // define roi here as x0, y0, width, height
	wCropped = u(roi);
	return 0;
}

int DirectAlignment::copyFlowColorToHost(cv::Mat &wCropped, float flowscale) {
	FlowToHSV(d_umed, width, height, stride, d_uvrgb, flowscale);
	//checkCudaErrors(cudaMemcpy((float *)depth.ptr(), d_w, stride * height * sizeof(float), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy((float3 *)uvrgb.ptr(), d_uvrgb, dataSize32fc3, cudaMemcpyDeviceToHost));
	cv::Rect roi(0, 0, width, height); // define roi here as x0, y0, width, height
	wCropped = uvrgb(roi);
	return 0;
}

int DirectAlignment::iAlignUp(int n)
{
	int m = this->StrideAlignment;
	int mod = n % m;

	if (mod)
		return n + m - mod;
	else
		return n;
}

int DirectAlignment::iDivUp(int n, int m)
{
	return (n + m - 1) / m;
}

template<typename T> void DirectAlignment::Swap(T &a, T &ax)
{
	T t = a;
	a = ax;
	ax = t;
}