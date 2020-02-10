#include "flow.h"

void FLOWDEBUGIMAGE(std::string windowName, float* deviceImage, int height, int stride, bool verbose, bool wait) {
	cv::Mat calibrated = cv::Mat(height, stride, CV_32F);
	checkCudaErrors(cudaMemcpy((float *)calibrated.ptr(), deviceImage, stride * height * sizeof(float), cudaMemcpyDeviceToHost));
	cv::imshow(windowName, calibrated);
	if (verbose) {
		std::cout << windowName << " " << calibrated.at<float>(height / 2, stride / 2) << std::endl;
	}
	if (wait) {
		cv::waitKey();
	}
	else {
		cv::waitKey(1);
	}

}

Flow::Flow() {
	this->BlockHeight = 1;
	this->BlockWidth = 32;
	this->StrideAlignment = 32;
}

int Flow::initialize(int width, int height, float lambda, float theta, float tau,
	int nLevels, float fScale, int nWarpIters, int nSolverIters) {

	this->width = width;
	this->height = height;
	this->stride = this->iAlignUp(width);
	this->lambda = lambda;
	this->theta = theta;
	this->tau = tau;
	this->fScale = fScale;
	this->nLevels = nLevels;
	this->nWarpIters = nWarpIters;
	this->nSolverIters = nSolverIters;

	pI0 = std::vector<float*>(nLevels);
	pI1 = std::vector<float*>(nLevels);
	pW = std::vector<int>(nLevels);
	pH = std::vector<int>(nLevels);
	pS = std::vector<int>(nLevels);
	pDataSize = std::vector<int>(nLevels);
	pMask = std::vector<float*>(nLevels);

	int newHeight = height;
	int newWidth = width;
	int newStride = iAlignUp(width);
	//std::cout << "Pyramid Sizes: " << newWidth << " " << newHeight << " " << newStride << std::endl;
	for (int level = 0; level < nLevels; level++) {
		pDataSize[level] = newStride * newHeight * sizeof(float);
		checkCudaErrors(cudaMalloc(&pI0[level], pDataSize[level]));
		checkCudaErrors(cudaMalloc(&pI1[level], pDataSize[level]));;
		checkCudaErrors(cudaMalloc(&pMask[level], pDataSize[level]));

		//std::cout << newHeight << " " << newWidth << " " << newStride << std::endl;

		pW[level] = newWidth;
		pH[level] = newHeight;
		pS[level] = newStride;
		newHeight = (int)((float)newHeight / fScale);
		newWidth = (int)((float)newWidth / fScale);
		newStride = iAlignUp(newWidth);
	}

	//std::cout << stride << " " << height << std::endl;
	dataSize8u = stride * height * sizeof(uchar);
	dataSize8uc3 = stride * height * sizeof(uchar3);
	dataSize32f = stride * height * sizeof(float);
	dataSize32fc2 = stride * height * sizeof(float2);
	dataSize32fc3 = stride * height * sizeof(float3);
	dataSize32fc4 = stride * height * sizeof(float4);

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

	// Output Optical Flow
	checkCudaErrors(cudaMalloc(&d_u, dataSize32fc2));
	checkCudaErrors(cudaMalloc(&d_us, dataSize32fc2));
	checkCudaErrors(cudaMalloc(&d_umed, dataSize32fc2));
	checkCudaErrors(cudaMalloc(&d_umeds, dataSize32fc2));

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

int Flow::copyImagesToDevice(cv::Mat i0, cv::Mat i1) {
	// Padding
	cv::copyMakeBorder(i0, im0pad, 0, 0, 0, stride - width, cv::BORDER_CONSTANT, 0);
	cv::copyMakeBorder(i1, im1pad, 0, 0, 0, stride - width, cv::BORDER_CONSTANT, 0);

	if (i0.type() == CV_8U) {
		checkCudaErrors(cudaMemcpy(d_i08u, (uchar *)im0pad.ptr(), dataSize8u, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_i18u, (uchar *)im1pad.ptr(), dataSize8u, cudaMemcpyHostToDevice));
		// Convert to 32F
		Cv8uToGray(d_i08u, pI0[0], width, height, stride);
		Cv8uToGray(d_i18u, pI1[0], width, height, stride);
	}
	else if (i0.type() == CV_32F) {
		checkCudaErrors(cudaMemcpy(pI0[0], (float *)im0pad.ptr(), dataSize32f, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(pI1[0], (float *)im1pad.ptr(), dataSize32f, cudaMemcpyHostToDevice));
	}
	else if (i0.type() == CV_8UC3) {
		checkCudaErrors(cudaMemcpy(d_i08uc3, (uchar3 *)im0pad.ptr(), dataSize8uc3, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_i18uc3, (uchar3 *)im1pad.ptr(), dataSize8uc3, cudaMemcpyHostToDevice));
		// Convert to 32F
		Cv8uc3ToGray(d_i08uc3, pI0[0], width, height, stride);
		Cv8uc3ToGray(d_i18uc3, pI1[0], width, height, stride);
	}
	return 0;
}

int Flow::copyMaskToDevice(cv::Mat mask) {
	cv::copyMakeBorder(mask, maskPad, 0, 0, 0, stride - width, cv::BORDER_CONSTANT, 0);
	checkCudaErrors(cudaMemcpy(pMask[0], (float *)maskPad.ptr(), dataSize32f, cudaMemcpyHostToDevice));
	for (int level = 1; level < nLevels; level++) {
		//std::cout << pW[level] << " " << pH[level] << " " << pS[level] << std::endl;
		DownscaleNearestNeighbor(pMask[level - 1], pW[level - 1], pH[level - 1], pS[level - 1],
			pW[level], pH[level], pS[level], pMask[level]);
		//DEBUGIMAGE("maskasdfadf", pFisheyeMask[level], pH[level], pS[level], true, true);
	}

	return 0;
}

int Flow::solveOpticalFlow() {
	checkCudaErrors(cudaMemset(d_u, 0, dataSize32fc2));
	checkCudaErrors(cudaMemset(d_umed, 0, dataSize32fc2));
	checkCudaErrors(cudaMemset(d_pu, 0, dataSize32fc2));
	checkCudaErrors(cudaMemset(d_pv, 0, dataSize32fc2));
	checkCudaErrors(cudaMemset(d_pus, 0, dataSize32fc2));
	checkCudaErrors(cudaMemset(d_pvs, 0, dataSize32fc2));

	// Construct pyramid
	for (int level = 1; level < nLevels; level++) {
		Downscale(pI0[level - 1], pW[level - 1], pH[level - 1], pS[level - 1],
			pW[level], pH[level], pS[level], pI0[level]);
		Downscale(pI1[level - 1], pW[level - 1], pH[level - 1], pS[level - 1],
			pW[level], pH[level], pS[level], pI1[level]);
	}

	// Solve TVL1 optical flow
	for (int level = nLevels - 1; level >= 0; level--) {
		for (int warpIter = 0; warpIter < nWarpIters; warpIter++) {
			checkCudaErrors(cudaMemset(d_pu, 0, dataSize32fc2));
			checkCudaErrors(cudaMemset(d_pv, 0, dataSize32fc2));
			//checkCudaErrors(cudaMemset(d_du, 0, dataSize32fc2));

			WarpImage(pI1[level], pW[level], pH[level], pS[level], d_u, d_i1warp);
			/*if (level == 0) {
				FLOWDEBUGIMAGE("warped", d_i1warp, pH[level], pS[level], false, false);
			}*/
			ComputeDerivatives(pI0[level], d_i1warp, pW[level], pH[level], pS[level], d_Ix, d_Iy, d_Iz);

			// Inner iteration
			for (int iter = 0; iter < nSolverIters; iter++) {
				// Solve Problem1A
				ThresholdingL1Masked(pMask[level], d_u, d_umed, d_Ix, d_Iy, d_Iz, lambda, theta, 
					pW[level], pH[level], pS[level]);
				//Swap(d_u, d_us);

				// Solve Problem1B
				SolveProblem1bMasked(pMask[level], d_u, d_pu, d_pv, theta, d_umed, 
					pW[level], pH[level], pS[level]);

				// Solve Problem2
				SolveProblem2Masked(pMask[level], d_umed, d_pu, d_pv, theta, tau, d_pus, d_pvs, 
					pW[level], pH[level], pS[level]);

				Swap(d_pu, d_pus);
				Swap(d_pv, d_pvs);
			}

			MedianFilter(d_u, pW[level], pH[level], pS[level], d_us, 5);
			Swap(d_u, d_us);
		}

		// Upscale
		if (level > 0)
		{
			float scale = fScale;
			Upscale(d_u, pW[level], pH[level], pS[level],
				pW[level - 1], pH[level - 1], pS[level - 1], scale, d_us);
			Upscale(d_umed, pW[level], pH[level], pS[level],
				pW[level - 1], pH[level - 1], pS[level - 1], scale, d_umeds);

			Swap(d_u, d_us);
			Swap(d_umed, d_umeds);
		}
	}

	return 0;
}

int Flow::copyFlowToHost(cv::Mat &wCropped) {
	// Remove Padding
	checkCudaErrors(cudaMemcpy((float2 *)u.ptr(), d_umed, dataSize32fc2, cudaMemcpyDeviceToHost));
	cv::Rect roi(0, 0, width, height); // define roi here as x0, y0, width, height
	wCropped = u(roi);
	return 0;
}

int Flow::copyFlowColorToHost(cv::Mat &wCropped, float flowscale) {
	FlowToHSV(d_umed, width, height, stride, d_uvrgb, flowscale);
	checkCudaErrors(cudaMemcpy((float3 *)uvrgb.ptr(), d_uvrgb, dataSize32fc3, cudaMemcpyDeviceToHost));
	cv::Rect roi(0, 0, width, height); // define roi here as x0, y0, width, height
	wCropped = uvrgb(roi);
	return 0;
}


// Utilities
int Flow::iAlignUp(int n)
{
	int m = this->StrideAlignment;
	int mod = n % m;

	if (mod)
		return n + m - mod;
	else
		return n;
}

int Flow::iDivUp(int n, int m)
{
	return (n + m - 1) / m;
}

template<typename T> void Flow::Swap(T &a, T &ax)
{
	T t = a;
	a = ax;
	ax = t;
}

template<typename T> void Flow::Copy(T &dst, T &src)
{
	dst = src;
}