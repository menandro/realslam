#include "stereolite.h"

void DEBUGWARPIMAGE(std::string windowName, float* deviceImage, int height, int stride, bool verbose, bool wait) {
	cv::Mat calibrated = cv::Mat(height, stride, CV_32F);
	checkCudaErrors(cudaMemcpy((float *)calibrated.ptr(), deviceImage, stride * height * sizeof(float), cudaMemcpyDeviceToHost));
	
	cv::imshow(windowName, calibrated);
	if (verbose) {
		std::cout << windowName << " " << calibrated.at<float>((int) ((float)height / 1.2f), stride / 2) << std::endl;
	}
	if (wait) {
		cv::waitKey();
	}
	else {
		cv::waitKey(1);
	}
}

void DEBUGDEPTH(std::string windowName, float* disparity, float baseline, float focal, int height, int stride, bool verbose, bool wait) {
	cv::Mat disparityMat = cv::Mat(height, stride, CV_32F);
	checkCudaErrors(cudaMemcpy((float *)disparityMat.ptr(), disparity, stride * height * sizeof(float), cudaMemcpyDeviceToHost));
	
	cv::Mat depth = baseline * focal / disparityMat;
	cv::Mat u_norm, u_gray, u_color;
	u_norm = depth * 256.0f / 5.0f;
	u_norm.convertTo(u_gray, CV_8UC1);
	cv::applyColorMap(u_gray, u_color, cv::COLORMAP_JET);

	cv::imshow(windowName, u_color);
	if (verbose) {
		std::cout << windowName << " " << disparityMat.at<float>(height / 2, stride / 2) << std::endl;
	}
	if (wait) {
		cv::waitKey();
	}
	else cv::waitKey(1);

}

StereoLite::StereoLite() {
	this->BlockHeight = 12;
	this->BlockWidth = 32;
	this->StrideAlignment = 32;
}

StereoLite::StereoLite(int blockWidth, int blockHeight, int strideAlignment) {
	this->BlockHeight = blockHeight;
	this->BlockWidth = blockWidth;
	this->StrideAlignment = strideAlignment;
}

int StereoLite::initialize(int width, int height, float lambda, float theta, float tau,
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
	pTvForward = std::vector<float2*>(nLevels);
	pTvBackward = std::vector<float2*>(nLevels);
	pFisheyeMask = std::vector<float*>(nLevels);
	pSparsePrior = std::vector<float*>(nLevels);

	int newHeight = height;
	int newWidth = width;
	int newStride = iAlignUp(width);
	//std::cout << "Pyramid Sizes: " << newWidth << " " << newHeight << " " << newStride << std::endl;
	for (int level = 0; level < nLevels; level++) {
		pDataSize[level] = newStride * newHeight * sizeof(float);
		checkCudaErrors(cudaMalloc(&pI0[level], pDataSize[level]));
		checkCudaErrors(cudaMalloc(&pI1[level], pDataSize[level]));
		checkCudaErrors(cudaMalloc(&pTvForward[level], 2 * pDataSize[level]));
		checkCudaErrors(cudaMalloc(&pTvBackward[level], 2 * pDataSize[level]));
		checkCudaErrors(cudaMalloc(&pFisheyeMask[level], pDataSize[level]));
		checkCudaErrors(cudaMalloc(&pSparsePrior[level], pDataSize[level]));

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
	checkCudaErrors(cudaMalloc(&d_Iu, dataSize32f));
	checkCudaErrors(cudaMalloc(&d_Iz, dataSize32f));
	// Output Disparity
	checkCudaErrors(cudaMalloc(&d_u, dataSize32f));
	checkCudaErrors(cudaMalloc(&d_du, dataSize32f));
	checkCudaErrors(cudaMalloc(&d_us, dataSize32f));
	// Output Depth
	checkCudaErrors(cudaMalloc(&d_depth, dataSize32f));
	// Warping Variables
	checkCudaErrors(cudaMalloc(&d_warpUV, dataSize32fc2));
	checkCudaErrors(cudaMalloc(&d_dwarpUV, dataSize32fc2));
	checkCudaErrors(cudaMalloc(&d_warpUVs, dataSize32fc2));

	// Vector Fields
	checkCudaErrors(cudaMalloc(&d_tvForward, dataSize32fc2));
	checkCudaErrors(cudaMalloc(&d_tvBackward, dataSize32fc2));
	checkCudaErrors(cudaMalloc(&d_tv2, dataSize32fc2));
	checkCudaErrors(cudaMalloc(&d_cv, dataSize32fc2));
	checkCudaErrors(cudaMalloc(&d_i1calibrated, dataSize32f));

	// Process variables
	checkCudaErrors(cudaMalloc(&d_p, dataSize32fc2));
	checkCudaErrors(cudaMalloc(&d_ps, dataSize32fc2));
	checkCudaErrors(cudaMalloc(&d_u_, dataSize32f));
	checkCudaErrors(cudaMalloc(&d_u_last, dataSize32f));
	checkCudaErrors(cudaMalloc(&d_u_s, dataSize32f));

	// PlaneSweep
	checkCudaErrors(cudaMalloc(&ps_i1warp, dataSize32f));
	checkCudaErrors(cudaMalloc(&ps_i1warps, dataSize32f));
	checkCudaErrors(cudaMalloc(&ps_error, dataSize32f));
	checkCudaErrors(cudaMalloc(&ps_depth, dataSize32f));
	checkCudaErrors(cudaMalloc(&ps_disparityForward, dataSize32f));
	checkCudaErrors(cudaMalloc(&ps_disparityBackward, dataSize32f));
	checkCudaErrors(cudaMalloc(&ps_disparityFinal, dataSize32f));
	checkCudaErrors(cudaMalloc(&ps_currentWarpForward, dataSize32fc2));
	checkCudaErrors(cudaMalloc(&ps_warpForward, dataSize32fc2));
	checkCudaErrors(cudaMalloc(&ps_finalWarpForward, dataSize32fc2));
	checkCudaErrors(cudaMalloc(&ps_grad, dataSize32f));
	checkCudaErrors(cudaMalloc(&ps_propagatedDisparity, dataSize32f));
	checkCudaErrors(cudaMalloc(&ps_propagatedWarpForward, dataSize32fc2));
	checkCudaErrors(cudaMalloc(&d_lambdaMask, dataSize32f));

	// 3D
	checkCudaErrors(cudaMalloc(&d_X, dataSize32fc3));

	// Debugging
	checkCudaErrors(cudaMalloc(&debug_depth, dataSize32f));
	checkCudaErrors(cudaMalloc(&d_uvrgb, dataSize32fc3));

	depth = cv::Mat(height, stride, CV_32F);
	warpUV = cv::Mat(height, stride, CV_32FC2);
	warpUVrgb = cv::Mat(height, stride, CV_32FC3);

	return 0;
}

int StereoLite::loadVectorFields(cv::Mat translationVector, cv::Mat calibrationVector) {
	// Padding
	cv::Mat translationVectorPad = cv::Mat(height, stride, CV_32FC2);
	cv::Mat calibrationVectorPad = cv::Mat(height, stride, CV_32FC2);
	cv::copyMakeBorder(translationVector, translationVectorPad, 0, 0, 0, stride - width, cv::BORDER_CONSTANT, 0);
	cv::copyMakeBorder(calibrationVector, calibrationVectorPad, 0, 0, 0, stride - width, cv::BORDER_CONSTANT, 0);

	// Translation Vector Field
	//translationVector = cv::Mat(height, stride, CV_32FC2);
	//calibrationVector = cv::Mat(height, stride, CV_32FC2);

	checkCudaErrors(cudaMemcpy(d_tvForward, (float2 *)translationVectorPad.ptr(), dataSize32fc2, cudaMemcpyHostToDevice));

	pTvForward[0] = d_tvForward;
	ScalarMultiply(d_tvForward, -1.0f, width, height, stride, d_tvBackward);
	pTvBackward[0] = d_tvBackward;
	for (int level = 1; level < nLevels; level++) {
		//std::cout << "vectorfields " << pW[level] << " " << pH[level] << " " << pS[level] << std::endl;
		Downscale(pTvForward[level - 1], pW[level - 1], pH[level - 1], pS[level - 1],
			pW[level], pH[level], pS[level], pTvForward[level]);
		Downscale(pTvBackward[level - 1], pW[level - 1], pH[level - 1], pS[level - 1],
			pW[level], pH[level], pS[level], pTvBackward[level]);
	}

	// Calibration Vector Field
	checkCudaErrors(cudaMemcpy(d_cv, (float2 *)calibrationVectorPad.ptr(), dataSize32fc2, cudaMemcpyHostToDevice));
	return 0;
}

int StereoLite::copyImagesToDevice(cv::Mat i0, cv::Mat i1) {
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

int StereoLite::copyMaskToDevice(cv::Mat mask) {
	cv::copyMakeBorder(mask, fisheyeMaskPad, 0, 0, 0, stride - width, cv::BORDER_CONSTANT, 0);
	checkCudaErrors(cudaMemcpy(pFisheyeMask[0], (float *)fisheyeMaskPad.ptr(), dataSize32f, cudaMemcpyHostToDevice));

	for (int level = 1; level < nLevels; level++) {
		//std::cout << pW[level] << " " << pH[level] << " " << pS[level] << std::endl;
		DownscaleNearestNeighbor(pFisheyeMask[level - 1], pW[level - 1], pH[level - 1], pS[level - 1],
			pW[level], pH[level], pS[level], pFisheyeMask[level]);
		//DEBUGIMAGE("maskasdfadf", pFisheyeMask[level], pH[level], pS[level], true, true);
	}
	return 0;
}

int StereoLite::solveStereoForwardMasked() {
	// Warp i1 using vector fields=
	WarpImage(pI1[0], width, height, stride, d_cv, d_i1calibrated);
	Swap(pI1[0], d_i1calibrated);

	checkCudaErrors(cudaMemset(d_u, 0, dataSize32f));
	checkCudaErrors(cudaMemset(d_u_, 0, dataSize32f));
	checkCudaErrors(cudaMemset(d_warpUV, 0, dataSize32fc2));

	// Construct pyramid
	for (int level = 1; level < nLevels; level++) {
		Downscale(pI0[level - 1], pW[level - 1], pH[level - 1], pS[level - 1],
			pW[level], pH[level], pS[level], pI0[level]);
		Downscale(pI1[level - 1], pW[level - 1], pH[level - 1], pS[level - 1],
			pW[level], pH[level], pS[level], pI1[level]);
	}

	// Solve stereo
	for (int level = nLevels - 1; level >= 0; level--) {
		if (level == nLevels - 1) {
			ComputeOpticalFlowVector(d_u, pTvForward[level], pW[level], pH[level], pS[level], d_warpUV);
		}

		for (int warpIter = 0; warpIter < nWarpIters; warpIter++) {
			checkCudaErrors(cudaMemset(d_p, 0, dataSize32fc2));
			checkCudaErrors(cudaMemset(d_du, 0, dataSize32f));

			FindWarpingVector(d_warpUV, pTvForward[level], pW[level], pH[level], pS[level], d_tv2);
			WarpImage(pI1[level], pW[level], pH[level], pS[level], d_warpUV, d_i1warp);

			ComputeDerivativesFisheye(pI0[level], d_i1warp, pTvForward[level],
				pW[level], pH[level], pS[level], d_Iu, d_Iz);

			Clone(d_u_last, pW[level], pH[level], pS[level], d_u_);

			// Inner iteration
			for (int iter = 0; iter < nSolverIters; iter++) {
				// Solve Problem1A
				ThresholdingL1Masked(pFisheyeMask[level], d_u, d_u_, d_Iu, d_Iz, lambda, theta, d_us, pW[level], pH[level], pS[level]);
				Swap(d_u, d_us);

				// Solve Problem1B
				SolveProblem1bMasked(pFisheyeMask[level], d_u, d_p, theta, d_u_, pW[level], pH[level], pS[level]);

				// Solve Problem2
				SolveProblem2Masked(pFisheyeMask[level], d_u_, d_p, theta, tau, d_ps, pW[level], pH[level], pS[level]);
				Swap(d_p, d_ps);
			}
			Subtract(d_u_, d_u_last, pW[level], pH[level], pS[level], d_du);
			LimitRange(d_du, limitRange, pW[level], pH[level], pS[level], d_du);
			Add(d_u_last, d_du, pW[level], pH[level], pS[level], d_u_);
			Clone(d_u, pW[level], pH[level], pS[level], d_u_);

			ComputeOpticalFlowVector(d_du, d_tv2, pW[level], pH[level], pS[level], d_dwarpUV);
			Add(d_warpUV, d_dwarpUV, pW[level], pH[level], pS[level], d_warpUV);
		}

		// Upscale
		if (level > 0)
		{
			float scale = fScale;
			Upscale(d_u, pW[level], pH[level], pS[level],
				pW[level - 1], pH[level - 1], pS[level - 1], scale, d_us);
			Upscale(d_u_, pW[level], pH[level], pS[level],
				pW[level - 1], pH[level - 1], pS[level - 1], scale, d_u_s);
			Upscale(d_warpUV, pW[level], pH[level], pS[level],
				pW[level - 1], pH[level - 1], pS[level - 1], scale, d_warpUVs);

			Swap(d_u, d_us);
			Swap(d_u_, d_u_s);
			Swap(d_warpUV, d_warpUVs);
		}
	}

	return 0;
}

int StereoLite::copyStereoToHost(cv::Mat &wCropped) {
	// Convert Disparity to Depth
	ConvertDisparityToDepth(d_u, baseline, focal, width, height, stride, d_depth);

	// Remove Padding
	//checkCudaErrors(cudaMemcpy((float *)depth.ptr(), d_w, stride * height * sizeof(float), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy((float *)depth.ptr(), d_depth, dataSize32f, cudaMemcpyDeviceToHost));
	cv::Rect roi(0, 0, width, height); // define roi here as x0, y0, width, height
	wCropped = depth(roi);
	return 0;
}


// PlaneSweep
int StereoLite::planeSweepForward() {
	// Plane sweep on level=1
	int planeSweepLevel = 0;
	checkCudaErrors(cudaMemset(ps_error, 0, dataSize32f));
	checkCudaErrors(cudaMemset(ps_depth, 0, dataSize32f));
	checkCudaErrors(cudaMemset(ps_disparityForward, 0, dataSize32f));
	Clone(ps_i1warp, pW[planeSweepLevel], pH[planeSweepLevel], pS[planeSweepLevel], pI1[planeSweepLevel]);
	SetValue(ps_error, planeSweepMaxError, pW[planeSweepLevel], pH[planeSweepLevel], pS[planeSweepLevel]);
	for (int sweep = 0; sweep < planeSweepMaxDisparity; sweep += planeSweepStride) {
		PlaneSweepCorrelation(ps_i1warp, pI0[planeSweepLevel], ps_disparityForward, sweep, planeSweepWindow,
			pW[planeSweepLevel], pH[planeSweepLevel], pS[planeSweepLevel], ps_error);
		for (int psStride = 0; psStride < planeSweepStride; psStride++) {
			WarpImage(ps_i1warp, pW[planeSweepLevel], pH[planeSweepLevel], pS[planeSweepLevel], d_tvForward, ps_i1warps);
			Swap(ps_i1warp, ps_i1warps);
		}
	}
	return 0;
}

int StereoLite::planeSweepBackward() {
	// Plane sweep on level=1
	int planeSweepLevel = 0;
	checkCudaErrors(cudaMemset(ps_error, 0, dataSize32f));
	checkCudaErrors(cudaMemset(ps_depth, 0, dataSize32f));
	checkCudaErrors(cudaMemset(ps_disparityBackward, 0, dataSize32f));
	Clone(ps_i1warp, pW[planeSweepLevel], pH[planeSweepLevel], pS[planeSweepLevel], pI0[planeSweepLevel]);
	SetValue(ps_error, planeSweepMaxError, pW[planeSweepLevel], pH[planeSweepLevel], pS[planeSweepLevel]);
	for (int sweep = 0; sweep < planeSweepMaxDisparity; sweep += planeSweepStride) {
		PlaneSweepCorrelation(ps_i1warp, pI1[planeSweepLevel], ps_disparityBackward, sweep, planeSweepWindow,
			pW[planeSweepLevel], pH[planeSweepLevel], pS[planeSweepLevel], ps_error);
		for (int psStride = 0; psStride < planeSweepStride; psStride++) {
			WarpImage(ps_i1warp, pW[planeSweepLevel], pH[planeSweepLevel], pS[planeSweepLevel], d_tvBackward, ps_i1warps);
			Swap(ps_i1warp, ps_i1warps);
		}
	}
	return 0;
}

int StereoLite::planeSweep() {
	// Forward
	int lvl = 0;
	WarpImage(pI1[0], width, height, stride, d_cv, d_i1calibrated);
	Swap(pI1[0], d_i1calibrated);

	checkCudaErrors(cudaMemset(ps_error, 0, dataSize32f));
	checkCudaErrors(cudaMemset(ps_depth, 0, dataSize32f));
	checkCudaErrors(cudaMemset(ps_disparityForward, 0, dataSize32f));
	checkCudaErrors(cudaMemset(ps_currentWarpForward, 0, dataSize32fc2));
	Clone(ps_i1warp, pW[lvl], pH[lvl], pS[lvl], pI1[lvl]);
	SetValue(ps_error, planeSweepMaxError, pW[lvl], pH[lvl], pS[lvl]);
	for (int sweep = 0; sweep < planeSweepMaxDisparity; sweep += planeSweepStride) {
		PlaneSweepCorrelationGetWarp(ps_i1warp, pI0[lvl], ps_disparityForward, sweep, planeSweepWindow,
			ps_currentWarpForward, ps_warpForward, d_tvForward, pW[lvl], pH[lvl], pS[lvl], ps_error);
		for (int psStride = 0; psStride < planeSweepStride; psStride++) {
			WarpImage(ps_i1warp, pW[lvl], pH[lvl], pS[lvl], d_tvForward, ps_i1warps);
			Swap(ps_i1warp, ps_i1warps);
		}
	}

	// Backward
	checkCudaErrors(cudaMemset(ps_error, 0, dataSize32f));
	checkCudaErrors(cudaMemset(ps_depth, 0, dataSize32f));
	checkCudaErrors(cudaMemset(ps_disparityBackward, 0, dataSize32f));
	Clone(ps_i1warp, pW[lvl], pH[lvl], pS[lvl], pI0[lvl]);
	SetValue(ps_error, planeSweepMaxError, pW[lvl], pH[lvl], pS[lvl]);
	for (int sweep = 0; sweep < planeSweepMaxDisparity; sweep += planeSweepStride) {
		PlaneSweepCorrelation(ps_i1warp, pI1[lvl], ps_disparityBackward, sweep, planeSweepWindow,
			pW[lvl], pH[lvl], pS[lvl], ps_error);
		for (int psStride = 0; psStride < planeSweepStride; psStride++) {
			WarpImage(ps_i1warp, pW[lvl], pH[lvl], pS[lvl], d_tvBackward, ps_i1warps);
			Swap(ps_i1warp, ps_i1warps);
		}
	}

	// Left-Right Consistency
	LeftRightConsistency(ps_disparityForward, ps_disparityBackward, ps_warpForward, planeSweepEpsilon,
		ps_disparityFinal, ps_finalWarpForward, pW[lvl], pH[lvl], pS[lvl]);

	return 0;
}

int StereoLite::copyPlanesweepForwardToHost(cv::Mat &wCropped) {
	// Convert Disparity to Depth
	ConvertDisparityToDepth(ps_disparityForward, baseline, focal, width, height, stride, d_depth);

	// Remove Padding
	//checkCudaErrors(cudaMemcpy((float *)depth.ptr(), d_w, stride * height * sizeof(float), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy((float *)depth.ptr(), d_depth, dataSize32f, cudaMemcpyDeviceToHost));
	cv::Rect roi(0, 0, width, height); // define roi here as x0, y0, width, height
	wCropped = depth(roi);
	return 0;
}

int StereoLite::copyPlanesweepBackwardToHost(cv::Mat &wCropped) {
	// Convert Disparity to Depth
	ConvertDisparityToDepth(ps_disparityBackward, baseline, focal, width, height, stride, d_depth);

	// Remove Padding
	//checkCudaErrors(cudaMemcpy((float *)depth.ptr(), d_w, stride * height * sizeof(float), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy((float *)depth.ptr(), d_depth, dataSize32f, cudaMemcpyDeviceToHost));
	cv::Rect roi(0, 0, width, height); // define roi here as x0, y0, width, height
	wCropped = depth(roi);
	return 0;
}

int StereoLite::copyPlanesweepFinalToHost(cv::Mat &wCropped) {
	// Convert Disparity to Depth
	ConvertDisparityToDepth(ps_disparityFinal, baseline, focal, width, height, stride, d_depth);

	// Remove Padding
	//checkCudaErrors(cudaMemcpy((float *)depth.ptr(), d_w, stride * height * sizeof(float), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy((float *)depth.ptr(), d_depth, dataSize32f, cudaMemcpyDeviceToHost));
	cv::Rect roi(0, 0, width, height); // define roi here as x0, y0, width, height
	wCropped = depth(roi);
	return 0;
}



// PlaneSweep with Propagation Only
int StereoLite::planeSweepWithPropagation(int upsamplingRadius) {
	planeSweep();
	int level = 0;
	Gradient(pI1[level], ps_grad);
	cudaMemset(ps_propagatedDisparity, 0, dataSize32f);
	for (int propIter = 0; propIter < maxPropIter; propIter++) {
		PropagateColorOnly(ps_grad, ps_disparityFinal, ps_finalWarpForward, ps_propagatedWarpForward,
			ps_propagatedDisparity, upsamplingRadius);
		Swap(ps_disparityFinal, ps_propagatedDisparity);
		Swap(ps_finalWarpForward, ps_propagatedWarpForward);
	}
	return 0;
}

// PlaneSweep with TVL1 Refinement
int StereoLite::planeSweepPropagationL1Refinement(int upsamplingRadius) {
	planeSweep();
	int level = 0;

	Gradient(pI1[level], ps_grad);
	cudaMemset(ps_propagatedDisparity, 0, dataSize32f);
	for (int propIter = 0; propIter < maxPropIter; propIter++) {
		PropagateColorOnly(ps_grad, ps_disparityFinal, ps_finalWarpForward, ps_propagatedWarpForward,
			ps_propagatedDisparity, upsamplingRadius);
		Swap(ps_disparityFinal, ps_propagatedDisparity);
		Swap(ps_finalWarpForward, ps_propagatedWarpForward);
	}
	

	Clone(d_warpUV, pW[level], pH[level], pS[level], ps_finalWarpForward);
	Clone(d_u_, pW[level], pH[level], pS[level], ps_disparityFinal);
	Clone(d_u, pW[level], pH[level], pS[level], ps_disparityFinal);

	for (int warpIter = 0; warpIter < nWarpIters; warpIter++) {
		checkCudaErrors(cudaMemset(d_p, 0, dataSize32fc2));
		checkCudaErrors(cudaMemset(d_du, 0, dataSize32f));

		FindWarpingVector(d_warpUV, pTvForward[level], pW[level], pH[level], pS[level], d_tv2);
		WarpImage(pI1[level], pW[level], pH[level], pS[level], d_warpUV, d_i1warp);

		ComputeDerivativesFisheye(pI0[level], d_i1warp, pTvForward[level],
			pW[level], pH[level], pS[level], d_Iu, d_Iz);

		Clone(d_u_last, pW[level], pH[level], pS[level], d_u_);

		// Inner iteration
		for (int iter = 0; iter < nSolverIters; iter++) {
			// Solve Problem1A
			ThresholdingL1Masked(pFisheyeMask[level], d_u, d_u_, d_Iu, d_Iz, lambda, theta, d_us, pW[level], pH[level], pS[level]);
			Swap(d_u, d_us);

			// Solve Problem1B
			SolveProblem1bMasked(pFisheyeMask[level], d_u, d_p, theta, d_u_, pW[level], pH[level], pS[level]);

			// Solve Problem2
			SolveProblem2Masked(pFisheyeMask[level], d_u_, d_p, theta, tau, d_ps, pW[level], pH[level], pS[level]);
			Swap(d_p, d_ps);
		}
		Subtract(d_u_, d_u_last, pW[level], pH[level], pS[level], d_du);
		LimitRange(d_du, limitRange, pW[level], pH[level], pS[level], d_du);
		Add(d_u_last, d_du, pW[level], pH[level], pS[level], d_u_);
		Clone(d_u, pW[level], pH[level], pS[level], d_u_);

		ComputeOpticalFlowVector(d_du, d_tv2, pW[level], pH[level], pS[level], d_dwarpUV);
		Add(d_warpUV, d_dwarpUV, pW[level], pH[level], pS[level], d_warpUV);

		////DEBUGDEPTH("depth", d_u, baseline, focal, pH[level], pS[level], false, false);
		//DEBUGWARPIMAGE("warp", d_i1warp, pH[level], pS[level], false, false);
		//cv::waitKey(1);
	}
	return 0;
}

int StereoLite::planeSweepL1upsamplingAndRefinement() {
	planeSweep();
	int level = 0;

	Clone(d_warpUV, pW[level], pH[level], pS[level], ps_finalWarpForward);
	Clone(d_u_, pW[level], pH[level], pS[level], ps_disparityFinal);
	Clone(d_u, pW[level], pH[level], pS[level], ps_disparityFinal);
	GetMask(ps_disparityFinal, d_lambdaMask, true, pW[level], pH[level], pS[level]);

	//// Run TVL1
	for (int warpIter = 0; warpIter < nWarpIters; warpIter++) {
		checkCudaErrors(cudaMemset(d_p, 0, dataSize32fc2));
		checkCudaErrors(cudaMemset(d_du, 0, dataSize32f));

		FindWarpingVector(d_warpUV, pTvForward[level], pW[level], pH[level], pS[level], d_tv2);
		WarpImage(pI1[level], pW[level], pH[level], pS[level], d_warpUV, d_i1warp);

		ComputeDerivativesFisheye(pI0[level], d_i1warp, pTvForward[level],
			pW[level], pH[level], pS[level], d_Iu, d_Iz);

		Clone(d_u_last, pW[level], pH[level], pS[level], d_u_);

		// Inner iteration
		for (int iter = 0; iter < nSolverIters; iter++) {
			// Solve Problem1A
			ThresholdingL1LambdaMasked(pFisheyeMask[level], d_u, d_u_, d_Iu, d_Iz, lambda, d_lambdaMask, theta, d_us, pW[level], pH[level], pS[level]);
			Swap(d_u, d_us);

			// Solve Problem1B
			SolveProblem1bMasked(pFisheyeMask[level], d_u, d_p, theta, d_u_, pW[level], pH[level], pS[level]);

			// Solve Problem2
			SolveProblem2Masked(pFisheyeMask[level], d_u_, d_p, theta, tau, d_ps, pW[level], pH[level], pS[level]);
			Swap(d_p, d_ps);

			DEBUGDEPTH("depth", d_u, baseline, focal, pH[level], pS[level], false, false);
			DEBUGWARPIMAGE("warp", d_i1warp, pH[level], pS[level], false, false);
			cv::waitKey(1);
		}

		//Subtract(d_u_, d_u_last, pW[level], pH[level], pS[level], d_du);
		//LimitRange(d_du, limitRange, pW[level], pH[level], pS[level], d_du);
		//Add(d_u_last, d_du, pW[level], pH[level], pS[level], d_u_);
		Clone(d_u, pW[level], pH[level], pS[level], d_u_);

		ComputeOpticalFlowVector(d_du, d_tv2, pW[level], pH[level], pS[level], d_dwarpUV);
		Add(d_warpUV, d_dwarpUV, pW[level], pH[level], pS[level], d_warpUV);
	}
	return 0;
}

int StereoLite::planeSweepL2upsamplingL1Refinement() {
	planeSweep();
	int level = 0;

	Clone(d_warpUV, pW[level], pH[level], pS[level], ps_finalWarpForward);
	Clone(d_u_, pW[level], pH[level], pS[level], ps_disparityFinal);
	Clone(d_u, pW[level], pH[level], pS[level], ps_disparityFinal);
	GetMask(ps_disparityFinal, d_lambdaMask, true, pW[level], pH[level], pS[level]);

	// One set of iteration to fill holes
	checkCudaErrors(cudaMemset(d_p, 0, dataSize32fc2));
	/*checkCudaErrors(cudaMemset(d_du, 0, dataSize32f));

	FindWarpingVector(d_warpUV, pTvForward[level], pW[level], pH[level], pS[level], d_tv2);
	WarpImage(pI1[level], pW[level], pH[level], pS[level], d_warpUV, d_i1warp);

	ComputeDerivativesFisheye(pI0[level], d_i1warp, pTvForward[level],
		pW[level], pH[level], pS[level], d_Iu, d_Iz);

	Clone(d_u_last, pW[level], pH[level], pS[level], d_u_);*/

	// Inner iteration
	for (int iter = 0; iter < l2iter; iter++) {
		// Solve Problem1A
		/*ThresholdingL1LambdaMasked(pFisheyeMask[level], d_u, d_u_, d_Iu, d_Iz, lambda, d_lambdaMask, 
			theta, d_us, pW[level], pH[level], pS[level]);
		Swap(d_u, d_us);*/
		SimpleL2(pFisheyeMask[level], d_u, d_u_, l2lambda, d_lambdaMask,
			d_us, pW[level], pH[level], pS[level]);
		Swap(d_u, d_us);

		// Solve Problem1B
		SolveProblem1bMasked(pFisheyeMask[level], d_u, d_p, theta, d_u_, pW[level], pH[level], pS[level]);

		// Solve Problem2
		SolveProblem2Masked(pFisheyeMask[level], d_u_, d_p, theta, tau, d_ps, pW[level], pH[level], pS[level]);
		Swap(d_p, d_ps);

		DEBUGDEPTH("depth", d_u, baseline, focal, pH[level], pS[level], false, false);
		DEBUGWARPIMAGE("warp", d_i1warp, pH[level], pS[level], false, false);
		cv::waitKey(1);
	}
	//Subtract(d_u_, d_u_last, pW[level], pH[level], pS[level], d_du);
	//LimitRange(d_du, limitRange, pW[level], pH[level], pS[level], d_du);
	//Add(d_u_last, d_du, pW[level], pH[level], pS[level], d_u_);
	//Clone(d_u, pW[level], pH[level], pS[level], d_u_);

	//ComputeOpticalFlowVector(d_du, d_tv2, pW[level], pH[level], pS[level], d_dwarpUV);
	//Add(d_warpUV, d_dwarpUV, pW[level], pH[level], pS[level], d_warpUV);

	//// Run TVL1
	//for (int warpIter = 0; warpIter < nWarpIters; warpIter++) {
	//	checkCudaErrors(cudaMemset(d_p, 0, dataSize32fc2));
	//	checkCudaErrors(cudaMemset(d_du, 0, dataSize32f));

	//	FindWarpingVector(d_warpUV, pTvForward[level], pW[level], pH[level], pS[level], d_tv2);
	//	WarpImage(pI1[level], pW[level], pH[level], pS[level], d_warpUV, d_i1warp);

	//	ComputeDerivativesFisheye(pI0[level], d_i1warp, pTvForward[level],
	//		pW[level], pH[level], pS[level], d_Iu, d_Iz);

	//	Clone(d_u_last, pW[level], pH[level], pS[level], d_u_);

	//	// Inner iteration
	//	for (int iter = 0; iter < nSolverIters; iter++) {
	//		// Solve Problem1A
	//		ThresholdingL1Masked(pFisheyeMask[level], d_u, d_u_, d_Iu, d_Iz, lambda, theta, d_us, pW[level], pH[level], pS[level]);
	//		Swap(d_u, d_us);

	//		// Solve Problem1B
	//		SolveProblem1bMasked(pFisheyeMask[level], d_u, d_p, theta, d_u_, pW[level], pH[level], pS[level]);

	//		// Solve Problem2
	//		SolveProblem2Masked(pFisheyeMask[level], d_u_, d_p, theta, tau, d_ps, pW[level], pH[level], pS[level]);
	//		Swap(d_p, d_ps);
	//	}
	//	Subtract(d_u_, d_u_last, pW[level], pH[level], pS[level], d_du);
	//	LimitRange(d_du, limitRange, pW[level], pH[level], pS[level], d_du);
	//	Add(d_u_last, d_du, pW[level], pH[level], pS[level], d_u_);
	//	Clone(d_u, pW[level], pH[level], pS[level], d_u_);

	//	ComputeOpticalFlowVector(d_du, d_tv2, pW[level], pH[level], pS[level], d_dwarpUV);
	//	Add(d_warpUV, d_dwarpUV, pW[level], pH[level], pS[level], d_warpUV);
	//}
	return 0;
}

int StereoLite::planeSweepPyramidL1Refinement() {
	planeSweep();
	Clone(pSparsePrior[0], pW[0], pH[0], pS[0], ps_disparityFinal);
	
	checkCudaErrors(cudaMemset(d_u, 0, dataSize32f));
	checkCudaErrors(cudaMemset(d_u_, 0, dataSize32f));
	checkCudaErrors(cudaMemset(d_warpUV, 0, dataSize32fc2));
	
	// Construct pyramid
	for (int level = 1; level < nLevels; level++) {
		Downscale(pI0[level - 1], pW[level - 1], pH[level - 1], pS[level - 1],
			pW[level], pH[level], pS[level], pI0[level]);
		Downscale(pI1[level - 1], pW[level - 1], pH[level - 1], pS[level - 1],
			pW[level], pH[level], pS[level], pI1[level]);
		// Downscale planesweep u and warpUV
		DownscaleNearestNeighbor(pSparsePrior[level - 1], pW[level - 1], pH[level - 1], pS[level - 1],
			pW[level], pH[level], pS[level], fScale, pSparsePrior[level]);
	}

	// Solve stereo
	for (int level = nLevels - 1; level >= 0; level--) {
		if (level == nLevels - 1) {
			ComputeOpticalFlowVector(d_u, pTvForward[level], pW[level], pH[level], pS[level], d_warpUV);
		}

		GetMask(pSparsePrior[level], d_lambdaMask, true, pW[level], pH[level], pS[level]);
		//SetValue(d_lambdaMask, 0.0f, pW[level], pH[level], pS[level]);
		//DEBUGWARPIMAGE("mask", d_lambdaMask, pH[level], pS[level], false, false);

		for (int warpIter = 0; warpIter < nWarpIters; warpIter++) {
			checkCudaErrors(cudaMemset(d_p, 0, dataSize32fc2));
			checkCudaErrors(cudaMemset(d_du, 0, dataSize32f));

			FindWarpingVector(d_warpUV, pTvForward[level], pW[level], pH[level], pS[level], d_tv2);
			WarpImage(pI1[level], pW[level], pH[level], pS[level], d_warpUV, d_i1warp);

			ComputeDerivativesFisheye(pI0[level], d_i1warp, pTvForward[level],
				pW[level], pH[level], pS[level], d_Iu, d_Iz);

			Clone(d_u_last, pW[level], pH[level], pS[level], d_u_);

			// Inner iteration
			int solverIters;
			if (level == 0) solverIters = nSolverIters;
			else solverIters = 2;
			
			for (int iter = 0; iter < solverIters; iter++) {
				// Solve Problem1A
				//ThresholdingL1Masked(pFisheyeMask[level], d_u, d_u_, d_Iu, d_Iz, lambda, theta, d_us, pW[level], pH[level], pS[level]);
				SparsePriorL1(pFisheyeMask[level], d_u, d_u_, pSparsePrior[level], d_Iu, d_Iz, 
					lambda, l2lambda, d_lambdaMask, theta, d_us, pW[level], pH[level], pS[level]);
				Swap(d_u, d_us);

				// Solve Problem1B
				SolveProblem1bMasked(pFisheyeMask[level], d_u, d_p, theta, d_u_, pW[level], pH[level], pS[level]);

				// Solve Problem2
				SolveProblem2Masked(pFisheyeMask[level], d_u_, d_p, theta, tau, d_ps, pW[level], pH[level], pS[level]);
				Swap(d_p, d_ps);

				//DEBUGDEPTH("imdepth", d_u_, baseline, focal, pH[level], pS[level], false, false);
				//DEBUGWARPIMAGE("warp", d_i1warp, pH[level], pS[level], false, false);
			}
			Subtract(d_u_, d_u_last, pW[level], pH[level], pS[level], d_du);
			LimitRange(d_du, limitRange, pW[level], pH[level], pS[level], d_du);
			Add(d_u_last, d_du, pW[level], pH[level], pS[level], d_u_);
			Clone(d_u, pW[level], pH[level], pS[level], d_u_);

			ComputeOpticalFlowVector(d_du, d_tv2, pW[level], pH[level], pS[level], d_dwarpUV);
			Add(d_warpUV, d_dwarpUV, pW[level], pH[level], pS[level], d_warpUV);
		}

		// Upscale
		if (level > 0)
		{
			float scale = fScale;
			Upscale(d_u, pW[level], pH[level], pS[level],
				pW[level - 1], pH[level - 1], pS[level - 1], scale, d_us);
			Upscale(d_u_, pW[level], pH[level], pS[level],
				pW[level - 1], pH[level - 1], pS[level - 1], scale, d_u_s);
			Upscale(d_warpUV, pW[level], pH[level], pS[level],
				pW[level - 1], pH[level - 1], pS[level - 1], scale, d_warpUVs);

			Swap(d_u, d_us);
			Swap(d_u_, d_u_s);
			Swap(d_warpUV, d_warpUVs);
		}
	}
	return 0;
}

int StereoLite::copyPropagatedDisparityToHost(cv::Mat &wCropped) {
	// Convert Disparity to Depth
	ConvertDisparityToDepth(ps_propagatedDisparity, baseline, focal, width, height, stride, d_depth);

	// Remove Padding
	//checkCudaErrors(cudaMemcpy((float *)depth.ptr(), d_w, stride * height * sizeof(float), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy((float *)depth.ptr(), d_depth, dataSize32f, cudaMemcpyDeviceToHost));
	cv::Rect roi(0, 0, width, height); // define roi here as x0, y0, width, height
	wCropped = depth(roi);
	return 0;
}

// Utilities
int StereoLite::iAlignUp(int n)
{
	int m = this->StrideAlignment;
	int mod = n % m;

	if (mod)
		return n + m - mod;
	else
		return n;
}

int StereoLite::iDivUp(int n, int m)
{
	return (n + m - 1) / m;
}

template<typename T> void StereoLite::Swap(T &a, T &ax)
{
	T t = a;
	a = ax;
	ax = t;
}

template<typename T> void StereoLite::Copy(T &dst, T &src)
{
	dst = src;
}