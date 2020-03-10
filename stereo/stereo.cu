#include "stereo.h"

Stereo::Stereo() {
	this->BlockWidth = 32;
	this->BlockHeight = 12;
	this->StrideAlignment = 32;
}

Stereo::Stereo(int BlockWidth, int BlockHeight, int StrideAlignment) {
	this->BlockWidth = BlockWidth;
	this->BlockHeight = BlockHeight;
	this->StrideAlignment = StrideAlignment;
}

int Stereo::initializeOpticalFlow(int width, int height, int channels, int inputType, int nLevels, float scale, float lambda,
	float theta, float tau, int nWarpIters, int nSolverIters)
{
	//allocate all memories
	this->width = width;
	this->height = height;
	this->stride = iAlignUp(width);
	this->inputType = inputType;

	this->fScale = scale;
	this->nLevels = nLevels;
	this->inputChannels = channels;
	this->nSolverIters = nSolverIters; //number of inner iteration (ROF loop)
	this->nWarpIters = nWarpIters;

	this->lambda = lambda;
	this->theta = theta;
	this->tau = tau;

	pI0 = std::vector<float*>(nLevels);
	pI1 = std::vector<float*>(nLevels);
	pW = std::vector<int>(nLevels);
	pH = std::vector<int>(nLevels);
	pS = std::vector<int>(nLevels);
	pDataSize = std::vector<int>(nLevels);

	int newHeight = height;
	int newWidth = width;
	int newStride = iAlignUp(width);
	//std::cout << "Pyramid Sizes: " << newWidth << " " << newHeight << " " << newStride << std::endl;
	for (int level = 0; level < nLevels; level++) {
		pDataSize[level] = newStride * newHeight * sizeof(float);
		checkCudaErrors(cudaMalloc(&pI0[level], pDataSize[level]));
		checkCudaErrors(cudaMalloc(&pI1[level], pDataSize[level]));
		
		pW[level] = newWidth;
		pH[level] = newHeight;
		pS[level] = newStride;
		newHeight = newHeight / fScale;
		newWidth = newWidth / fScale;
		newStride = iAlignUp(newWidth);
	}
	
	//runtime
	dataSize = stride * height * sizeof(float);
	dataSize8uc3 = stride * height * sizeof(uchar3);
	dataSize8u = stride * height * sizeof(uchar);
	dataSize32f = dataSize;
	dataSize32fc3 = dataSize * 3;
	checkCudaErrors(cudaMalloc(&d_i1warp, dataSize));

	checkCudaErrors(cudaMalloc(&d_du, dataSize));
	checkCudaErrors(cudaMalloc(&d_dv, dataSize));
	checkCudaErrors(cudaMalloc(&d_dus, dataSize));
	checkCudaErrors(cudaMalloc(&d_dvs, dataSize));

	checkCudaErrors(cudaMalloc(&d_dumed, dataSize));
	checkCudaErrors(cudaMalloc(&d_dvmed, dataSize));
	checkCudaErrors(cudaMalloc(&d_dumeds, dataSize));
	checkCudaErrors(cudaMalloc(&d_dvmeds, dataSize));

	//dual TV
	checkCudaErrors(cudaMalloc(&d_pu1, dataSize));
	checkCudaErrors(cudaMalloc(&d_pu2, dataSize));
	checkCudaErrors(cudaMalloc(&d_pv1, dataSize));
	checkCudaErrors(cudaMalloc(&d_pv2, dataSize));
	//dual TV temps
	checkCudaErrors(cudaMalloc(&d_pu1s, dataSize));
	checkCudaErrors(cudaMalloc(&d_pu2s, dataSize));
	checkCudaErrors(cudaMalloc(&d_pv1s, dataSize));
	checkCudaErrors(cudaMalloc(&d_pv2s, dataSize));

	checkCudaErrors(cudaMalloc(&d_Ix, dataSize));
	checkCudaErrors(cudaMalloc(&d_Iy, dataSize));
	checkCudaErrors(cudaMalloc(&d_Iz, dataSize));

	checkCudaErrors(cudaMalloc(&d_u, dataSize));
	checkCudaErrors(cudaMalloc(&d_v, dataSize));
	checkCudaErrors(cudaMalloc(&d_us, dataSize));
	checkCudaErrors(cudaMalloc(&d_vs, dataSize));

	if (inputType == CV_8UC3) {
		checkCudaErrors(cudaMalloc(&d_i08uc3, dataSize8uc3));
		checkCudaErrors(cudaMalloc(&d_i18uc3, dataSize8uc3));
	}
	else if (inputType == CV_8U) {
		checkCudaErrors(cudaMalloc(&d_i08u, dataSize8u));
		checkCudaErrors(cudaMalloc(&d_i18u, dataSize8u));
	}

	// colored uv, for display only
	checkCudaErrors(cudaMalloc(&d_uvrgb, dataSize * 3));

	// Output mats
	uvrgb = cv::Mat(height, stride, CV_32FC3);
	upad = cv::Mat(height, stride, CV_32F);
	vpad = cv::Mat(height, stride, CV_32F);

	return 0;
}

int Stereo::initializeFisheyeStereo(int width, int height, int channels, int inputType, int nLevels, float scale, float lambda,
	float theta, float tau, int nWarpIters, int nSolverIters) {
	//allocate all memories
	this->width = width;
	this->height = height;
	this->stride = iAlignUp(width);
	this->inputType = inputType;

	this->fScale = scale;
	this->nLevels = nLevels;
	this->inputChannels = channels;
	this->nSolverIters = nSolverIters; //number of inner iteration (ROF loop)
	this->nWarpIters = nWarpIters;

	this->lambda = lambda;
	this->theta = theta;
	this->tau = tau;

	pI0 = std::vector<float*>(nLevels);
	pI1 = std::vector<float*>(nLevels);
	pW = std::vector<int>(nLevels);
	pH = std::vector<int>(nLevels);
	pS = std::vector<int>(nLevels);
	pDataSize = std::vector<int>(nLevels);
	pTvxForward = std::vector<float*>(nLevels);
	pTvyForward = std::vector<float*>(nLevels);
	pTvxBackward = std::vector<float*>(nLevels);
	pTvyBackward = std::vector<float*>(nLevels);

	int newHeight = height;
	int newWidth = width;
	int newStride = iAlignUp(width);
	//std::cout << "Pyramid Sizes: " << newWidth << " " << newHeight << " " << newStride << std::endl;
	for (int level = 0; level < nLevels; level++) {
		pDataSize[level] = newStride * newHeight * sizeof(float);
		checkCudaErrors(cudaMalloc(&pI0[level], pDataSize[level]));
		checkCudaErrors(cudaMalloc(&pI1[level], pDataSize[level]));
		checkCudaErrors(cudaMalloc(&pTvxForward[level], pDataSize[level]));
		checkCudaErrors(cudaMalloc(&pTvyForward[level], pDataSize[level]));
		checkCudaErrors(cudaMalloc(&pTvxBackward[level], pDataSize[level]));
		checkCudaErrors(cudaMalloc(&pTvyBackward[level], pDataSize[level]));

		pW[level] = newWidth;
		pH[level] = newHeight;
		pS[level] = newStride;
		newHeight = newHeight / fScale;
		newWidth = newWidth / fScale;
		newStride = iAlignUp(newWidth);
	}

	//runtime
	dataSize = stride * height * sizeof(float);
	dataSize8uc3 = stride * height * sizeof(uchar3);
	dataSize8u = stride * height * sizeof(uchar);
	dataSize32f = dataSize;
	dataSize32fc3 = dataSize * 3;
	checkCudaErrors(cudaMalloc(&d_i1warp, dataSize));

	checkCudaErrors(cudaMalloc(&d_tvxForward, dataSize));
	checkCudaErrors(cudaMalloc(&d_tvyForward, dataSize));
	checkCudaErrors(cudaMalloc(&d_tvxBackward, dataSize));
	checkCudaErrors(cudaMalloc(&d_tvyBackward, dataSize));
	checkCudaErrors(cudaMalloc(&d_tvx2, dataSize));
	checkCudaErrors(cudaMalloc(&d_tvy2, dataSize));
	checkCudaErrors(cudaMalloc(&d_cvx, dataSize));
	checkCudaErrors(cudaMalloc(&d_cvy, dataSize));

	checkCudaErrors(cudaMalloc(&d_i1calibrated, dataSize));
	checkCudaErrors(cudaMalloc(&d_Iw, dataSize));
	checkCudaErrors(cudaMalloc(&d_Iz, dataSize));

	checkCudaErrors(cudaMalloc(&d_w, dataSize));
	checkCudaErrors(cudaMalloc(&d_wForward, dataSize));
	checkCudaErrors(cudaMalloc(&d_wBackward, dataSize));
	checkCudaErrors(cudaMalloc(&d_wFinal, dataSize));
	checkCudaErrors(cudaMalloc(&d_u, dataSize));
	checkCudaErrors(cudaMalloc(&d_v, dataSize));
	checkCudaErrors(cudaMalloc(&d_uForward, dataSize));
	checkCudaErrors(cudaMalloc(&d_vForward, dataSize));
	checkCudaErrors(cudaMalloc(&d_us, dataSize));
	checkCudaErrors(cudaMalloc(&d_vs, dataSize));
	checkCudaErrors(cudaMalloc(&d_ws, dataSize));

	checkCudaErrors(cudaMalloc(&d_du, dataSize));
	checkCudaErrors(cudaMalloc(&d_dv, dataSize));
	checkCudaErrors(cudaMalloc(&d_dw, dataSize));
	checkCudaErrors(cudaMalloc(&d_dws, dataSize));
	checkCudaErrors(cudaMalloc(&d_depth, dataSize));
	checkCudaErrors(cudaMalloc(&d_depthFinal, dataSize));
	checkCudaErrors(cudaMalloc(&d_occlusion, dataSize));

	checkCudaErrors(cudaMalloc(&d_dwmed, dataSize));
	checkCudaErrors(cudaMalloc(&d_dwmeds, dataSize));
	checkCudaErrors(cudaMalloc(&d_pw1, dataSize));
	checkCudaErrors(cudaMalloc(&d_pw2, dataSize));
	checkCudaErrors(cudaMalloc(&d_pw1s, dataSize));
	checkCudaErrors(cudaMalloc(&d_pw2s, dataSize));

	if (inputType == CV_8UC3) {
		checkCudaErrors(cudaMalloc(&d_i08uc3, dataSize8uc3));
		checkCudaErrors(cudaMalloc(&d_i18uc3, dataSize8uc3));
	}
	else if (inputType == CV_8U) {
		checkCudaErrors(cudaMalloc(&d_i08u, dataSize8u));
		checkCudaErrors(cudaMalloc(&d_i18u, dataSize8u));
	}

	// Plane sweep
	checkCudaErrors(cudaMalloc(&ps_i1warp, dataSize));
	checkCudaErrors(cudaMalloc(&ps_i1warps, dataSize));
	checkCudaErrors(cudaMalloc(&ps_error, dataSize));
	checkCudaErrors(cudaMalloc(&ps_depth, dataSize));
	checkCudaErrors(cudaMalloc(&ps_disparity, dataSize));
	checkCudaErrors(cudaMalloc(&ps_disparityForward, dataSize));
	checkCudaErrors(cudaMalloc(&ps_disparityBackward, dataSize));
	checkCudaErrors(cudaMalloc(&ps_disparityFinal, dataSize));

	// Colored uv, for display only
	checkCudaErrors(cudaMalloc(&d_uvrgb, dataSize * 3));
	uvrgb = cv::Mat(height, stride, CV_32FC3);
	disparity = cv::Mat(height, stride, CV_32F);
	depth = cv::Mat(height, stride, CV_32F);
	planeSweepDepth = cv::Mat(height, stride, CV_32F);
	return 0;
}

int Stereo::loadVectorFields(cv::Mat translationVector, cv::Mat calibrationVector) {
	// Padding
	cv::Mat translationVectorPad = cv::Mat(height, stride, CV_32F);
	cv::Mat calibrationVectorPad = cv::Mat(height, stride, CV_32F);
	cv::copyMakeBorder(translationVector, translationVectorPad, 0, 0, 0, stride - width, cv::BORDER_CONSTANT, 0);
	cv::copyMakeBorder(calibrationVector, calibrationVectorPad, 0, 0, 0, stride - width, cv::BORDER_CONSTANT, 0);

	// Translation Vector Field
	translationVectorX = cv::Mat(height, stride, CV_32F);
	translationVectorY = cv::Mat(height, stride, CV_32F);
	calibrationVectorX = cv::Mat(height, stride, CV_32F);
	calibrationVectorY = cv::Mat(height, stride, CV_32F);

	cv::Mat tuv[2];
	cv::split(translationVectorPad, tuv);
	translationVectorX = tuv[0];
	translationVectorY = tuv[1];
	checkCudaErrors(cudaMemcpy(d_tvxForward, (float *)translationVectorX.ptr(), dataSize32f, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_tvyForward, (float *)translationVectorY.ptr(), dataSize32f, cudaMemcpyHostToDevice));

	pTvxForward[0] = d_tvxForward;
	pTvyForward[0] = d_tvyForward;
	ScalarMultiply(d_tvxForward, -1.0f, width, height, stride, d_tvxBackward);
	ScalarMultiply(d_tvyForward, -1.0f, width, height, stride, d_tvyBackward);
	pTvxBackward[0] = d_tvxBackward;
	pTvyBackward[0] = d_tvyBackward;
	for (int level = 1; level < nLevels; level++) {
		//std::cout << pW[level] << " " << pH[level] << " " << pS[level] << std::endl;
		Downscale(pTvxForward[level - 1], pW[level - 1], pH[level - 1], pS[level - 1],
			pW[level], pH[level], pS[level], pTvxForward[level]);
		Downscale(pTvyForward[level - 1], pW[level - 1], pH[level - 1], pS[level - 1],
			pW[level], pH[level], pS[level], pTvyForward[level]);

		Downscale(pTvxBackward[level - 1], pW[level - 1], pH[level - 1], pS[level - 1],
			pW[level], pH[level], pS[level], pTvxBackward[level]);
		Downscale(pTvyBackward[level - 1], pW[level - 1], pH[level - 1], pS[level - 1],
			pW[level], pH[level], pS[level], pTvyBackward[level]);

	}

	// Calibration Vector Field
	cv::Mat cuv[2];
	cv::split(calibrationVectorPad, cuv);
	calibrationVectorX = cuv[0].clone();
	calibrationVectorY = cuv[1].clone();
	checkCudaErrors(cudaMemcpy(d_cvx, (float *)calibrationVectorX.ptr(), dataSize32f, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_cvy, (float *)calibrationVectorY.ptr(), dataSize32f, cudaMemcpyHostToDevice));
	return 0;
}

int Stereo::copyImagesToDevice(cv::Mat i0, cv::Mat i1) {
	// Padding
	cv::copyMakeBorder(i0, im0pad, 0, 0, 0, stride - width, cv::BORDER_CONSTANT, 0);
	cv::copyMakeBorder(i1, im1pad, 0, 0, 0, stride - width, cv::BORDER_CONSTANT, 0);

	if (inputType == CV_8U) {
		checkCudaErrors(cudaMemcpy(d_i08u, (uchar *)im0pad.ptr(), dataSize8u, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_i18u, (uchar *)im1pad.ptr(), dataSize8u, cudaMemcpyHostToDevice));
		// Convert to 32F
		Cv8uToGray(d_i08u, pI0[0], width, height, stride);
		Cv8uToGray(d_i18u, pI1[0], width, height, stride);
	}
	else if (inputType == CV_32F) {
		checkCudaErrors(cudaMemcpy(pI0[0], (float *)im0pad.ptr(), dataSize32f, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(pI1[0], (float *)im1pad.ptr(), dataSize32f, cudaMemcpyHostToDevice));
	}
	else {
		checkCudaErrors(cudaMemcpy(d_i08uc3, (uchar3 *)im0pad.ptr(), dataSize8uc3, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_i18uc3, (uchar3 *)im1pad.ptr(), dataSize8uc3, cudaMemcpyHostToDevice));
		// Convert to 32F
		Cv8uc3ToGray(d_i08uc3, pI0[0], width, height, stride);
		Cv8uc3ToGray(d_i18uc3, pI1[0], width, height, stride);
	}
	return 0;
}

int Stereo::solveStereoForward() {
	// Warp i1 using vector fields
	WarpImage(pI1[0], width, height, stride, d_cvx, d_cvy, d_i1calibrated);
	Swap(pI1[0], d_i1calibrated);

	checkCudaErrors(cudaMemset(d_w, 0, dataSize));
	checkCudaErrors(cudaMemset(d_u, 0, dataSize));
	checkCudaErrors(cudaMemset(d_v, 0, dataSize));
	// Construct pyramid
	for (int level = 1; level < nLevels; level++) {
		Downscale(pI0[level - 1],
			pW[level - 1], pH[level - 1], pS[level - 1],
			pW[level], pH[level], pS[level],
			pI0[level]);

		Downscale(pI1[level - 1],
			pW[level - 1], pH[level - 1], pS[level - 1],
			pW[level], pH[level], pS[level],
			pI1[level]);
	}

	//planeSweepForward();

	ComputeOpticalFlowVector(d_w, d_tvxForward, d_tvyForward, pW[0], pH[0], pS[0], d_u, d_v);

	/*cv::Mat calibrated = cv::Mat(height, stride, CV_32F);
	checkCudaErrors(cudaMemcpy((float *)calibrated.ptr(), ps_disparity, width * height * sizeof(float), cudaMemcpyDeviceToHost));
	cv::imshow("calibrated", calibrated/(float)planeSweepMaxDisparity);*/

	// Solve stereo
	for (int level = nLevels - 1; level >= 0; level--) {
		for (int warpIter = 0; warpIter < nWarpIters; warpIter++) {
			// Compute U,V from W d_w is magnitude of vector d_tvx, d_tvy
			// Warp using U,V
			//std::cout << "entered" << std::endl;
			checkCudaErrors(cudaMemset(d_du, 0, dataSize));
			checkCudaErrors(cudaMemset(d_dv, 0, dataSize));
			checkCudaErrors(cudaMemset(d_dw, 0, dataSize));

			checkCudaErrors(cudaMemset(d_dws, 0, dataSize));
			checkCudaErrors(cudaMemset(d_dwmed, 0, dataSize));
			checkCudaErrors(cudaMemset(d_dwmeds, 0, dataSize));
			checkCudaErrors(cudaMemset(d_pw1, 0, dataSize));
			checkCudaErrors(cudaMemset(d_pw2, 0, dataSize));

			FindWarpingVector(d_u, d_v, pTvxForward[level], pTvyForward[level], pW[level], pH[level], pS[level], d_tvx2, d_tvy2);
			WarpImage(pI1[level], pW[level], pH[level], pS[level], d_u, d_v, d_i1warp);
			//std::cout << pW[level] << " " << pH[level] << " " << pS[level] << std::endl;
			ComputeDerivativesFisheye(pI0[level], d_i1warp, pTvxForward[level], pTvyForward[level], pW[level], pH[level], pS[level], d_Iw, d_Iz);
			/*if (level == 0) {
				cv::Mat calibrated = cv::Mat(pH[level], pS[level], CV_32F);
				checkCudaErrors(cudaMemcpy((float *)calibrated.ptr(), d_i1warp, pS[level] * pH[level] * sizeof(float), cudaMemcpyDeviceToHost));
				cv::imshow("gradient", calibrated);
			}*/

			// Inner iteration
			for (int iter = 0; iter < nSolverIters; ++iter)
			{
				SolveDataL1Stereo(d_dwmed,
					d_pw1, d_pw2,
					d_Iw, d_Iz,
					pW[level], pH[level], pS[level],
					lambda, theta,
					d_dwmeds); //du1 = duhat output
				Swap(d_dwmed, d_dwmeds);

				SolveSmoothDualTVGlobalStereo(d_dwmed,
					d_pw1, d_pw2,
					pW[level], pH[level], pS[level],
					tau, theta,
					d_pw1s, d_pw2s);
				Swap(d_pw1, d_pw1s);
				Swap(d_pw2, d_pw2s);
			}

			// Sanity Check: Limit disparity to 1
			LimitRange(d_dwmed, 1.0f, pW[level], pH[level], pS[level], d_dwmeds);
			Swap(d_dwmed, d_dwmeds);

			//// One median filtering
			MedianFilterDisparity(d_dwmed, pW[level], pH[level], pS[level],
				d_dwmeds, 5);
			Swap(d_dwmed, d_dwmeds);

			//// Calculate d_du, d_dv
			ComputeOpticalFlowVector(d_dwmed, d_tvx2, d_tvy2, pW[level], pH[level], pS[level], d_du, d_dv);

			//// update w, u, v
			Add(d_w, d_dwmed, pH[level] * pS[level], d_w);
			Add(d_u, d_du, pH[level] * pS[level], d_u);
			Add(d_v, d_dv, pH[level] * pS[level], d_v);
		}

		// Upscale
		if (level > 0)
		{
			float scale = fScale;
			Upscale(d_u, pW[level], pH[level], pS[level], pW[level - 1], pH[level - 1], pS[level - 1], scale, d_us);
			Upscale(d_v, pW[level], pH[level], pS[level], pW[level - 1], pH[level - 1], pS[level - 1], scale, d_vs);
			Upscale(d_w, pW[level], pH[level], pS[level], pW[level - 1], pH[level - 1], pS[level - 1], scale, d_ws);
			Swap(d_u, d_us);
			Swap(d_v, d_vs);
			Swap(d_w, d_ws);
		}
	}

	Clone(d_w, width, height, stride, d_wForward);

	if (visualizeResults) {
		FlowToHSV(d_u, d_v, width, height, stride, d_uvrgb, flowScale);
	}

	return 0;
}

int Stereo::solveStereoBackward() {
	// Warp i1 using vector fields
	//WarpImage(pI1[0], width, height, stride, d_cvx, d_cvy, d_i1calibrated);
	//Swap(pI1[0], d_i1calibrated);
	Swap(pI0[0], pI1[0]);
	
	checkCudaErrors(cudaMemset(d_w, 0, dataSize));
	checkCudaErrors(cudaMemset(d_u, 0, dataSize));
	checkCudaErrors(cudaMemset(d_v, 0, dataSize));
	// Construct pyramid
	for (int level = 1; level < nLevels; level++) {
		Swap(pI0[level], pI1[level]);
	}

	//planeSweepBackward();

	//Clone(ps_disparity, pW[planeSweepLevel], pH[planeSweepLevel], pS[planeSweepLevel], d_w);
	ComputeOpticalFlowVector(d_w, d_tvxBackward, d_tvyBackward, pW[0], pH[0], pS[0], d_u, d_v);

	/*cv::Mat calibrated = cv::Mat(height, stride, CV_32F);
	checkCudaErrors(cudaMemcpy((float *)calibrated.ptr(), ps_disparity, width * height * sizeof(float), cudaMemcpyDeviceToHost));
	cv::imshow("calibrated", calibrated/(float)planeSweepMaxDisparity);*/

	// Solve stereo
	for (int level = nLevels - 1; level >= 0; level--) {
		for (int warpIter = 0; warpIter < nWarpIters; warpIter++) {
			// Compute U,V from W d_w is magnitude of vector d_tvx, d_tvy
			// Warp using U,V
			//std::cout << "entered" << std::endl;
			checkCudaErrors(cudaMemset(d_du, 0, dataSize));
			checkCudaErrors(cudaMemset(d_dv, 0, dataSize));
			checkCudaErrors(cudaMemset(d_dw, 0, dataSize));

			checkCudaErrors(cudaMemset(d_dws, 0, dataSize));
			checkCudaErrors(cudaMemset(d_dwmed, 0, dataSize));
			checkCudaErrors(cudaMemset(d_dwmeds, 0, dataSize));
			checkCudaErrors(cudaMemset(d_pw1, 0, dataSize));
			checkCudaErrors(cudaMemset(d_pw2, 0, dataSize));

			FindWarpingVector(d_u, d_v, pTvxBackward[level], pTvyBackward[level], 
				pW[level], pH[level], pS[level], d_tvx2, d_tvy2);
			WarpImage(pI1[level], pW[level], pH[level], pS[level], d_u, d_v, d_i1warp);
			//std::cout << pW[level] << " " << pH[level] << " " << pS[level] << std::endl;
			ComputeDerivativesFisheye(pI0[level], d_i1warp, pTvxBackward[level], pTvyBackward[level], 
				pW[level], pH[level], pS[level], d_Iw, d_Iz);
			/*if (level == 0) {
				cv::Mat calibrated = cv::Mat(pH[level], pS[level], CV_32F);
				checkCudaErrors(cudaMemcpy((float *)calibrated.ptr(), d_i1warp, pS[level] * pH[level] * sizeof(float), cudaMemcpyDeviceToHost));
				cv::imshow("gradient", calibrated);
			}*/

			// Inner iteration
			for (int iter = 0; iter < nSolverIters; ++iter)
			{
				SolveDataL1Stereo(d_dwmed, 
					d_pw1, d_pw2,
					d_Iw, d_Iz,
					pW[level], pH[level], pS[level],
					lambda, theta,
					d_dwmeds); //du1 = duhat output
				Swap(d_dwmed, d_dwmeds);

				SolveSmoothDualTVGlobalStereo(d_dwmed, 
					d_pw1, d_pw2,
					pW[level], pH[level], pS[level],
					tau, theta,
					d_pw1s, d_pw2s);
				Swap(d_pw1, d_pw1s);
				Swap(d_pw2, d_pw2s);
			}

			// Sanity Check: Limit disparity to 1
			LimitRange(d_dwmed, 1.0f, pW[level], pH[level], pS[level], d_dwmeds);
			Swap(d_dwmed, d_dwmeds);

			//// One median filtering
			MedianFilterDisparity(d_dwmed, pW[level], pH[level], pS[level],
				d_dwmeds, 5);
			Swap(d_dwmed, d_dwmeds);

			//// Calculate d_du, d_dv
			ComputeOpticalFlowVector(d_dwmed, d_tvx2, d_tvy2, pW[level], pH[level], pS[level], d_du, d_dv);

			//// update w, u, v
			Add(d_w, d_dwmed, pH[level] * pS[level], d_w);
			Add(d_u, d_du, pH[level] * pS[level], d_u);
			Add(d_v, d_dv, pH[level] * pS[level], d_v);
		}

		// Upscale
		if (level > 0)
		{
			float scale = fScale;
			Upscale(d_u, pW[level], pH[level], pS[level], pW[level - 1], pH[level - 1], pS[level - 1], scale, d_us);
			Upscale(d_v, pW[level], pH[level], pS[level], pW[level - 1], pH[level - 1], pS[level - 1], scale, d_vs);
			Upscale(d_w, pW[level], pH[level], pS[level], pW[level - 1], pH[level - 1], pS[level - 1], scale, d_ws);
			Swap(d_u, d_us);
			Swap(d_v, d_vs);
			Swap(d_w, d_ws);
		}
	}

	Clone(d_w, width, height, stride, d_wBackward);

	if (visualizeResults) {
		FlowToHSV(d_u, d_v, width, height, stride, d_uvrgb, flowScale);
	}

	return 0;
}

int Stereo::occlusionCheck(float threshold) {
	isOcclusionChecked = true;
	// Get wFinal
	OcclusionCheck(d_wForward, d_wBackward, threshold, d_uForward, d_vForward, width, height, stride, d_wFinal);
	return 0;
}

int Stereo::planeSweepForward() {
	// Plane sweep on level=1
	int planeSweepLevel = 0;
	checkCudaErrors(cudaMemset(ps_error, 0, dataSize));
	checkCudaErrors(cudaMemset(ps_depth, 0, dataSize));
	checkCudaErrors(cudaMemset(ps_disparity, 0, dataSize));
	Clone(pI1[planeSweepLevel], pW[planeSweepLevel], pH[planeSweepLevel], pS[planeSweepLevel], ps_i1warp);
	SetValue(ps_error, planeSweepMaxError, pW[planeSweepLevel], pH[planeSweepLevel], pS[planeSweepLevel]);
	for (int sweep = 0; sweep < planeSweepMaxDisparity; sweep += planeSweepStride) {
		PlaneSweepCorrelation(ps_i1warp, pI0[planeSweepLevel], ps_disparity, sweep, planeSweepWindow,
			pW[planeSweepLevel], pH[planeSweepLevel], pS[planeSweepLevel], ps_error);
		for (int psStride = 0; psStride < planeSweepStride; psStride++) {
			WarpImage(ps_i1warp, pW[planeSweepLevel], pH[planeSweepLevel], pS[planeSweepLevel], d_tvxForward, d_tvyForward, ps_i1warps);
			Swap(ps_i1warp, ps_i1warps);
		}
	}
	//Clone(ps_disparity, pW[planeSweepLevel], pH[planeSweepLevel], pS[planeSweepLevel], d_w);
	return 0;
}

int Stereo::planeSweepBackward() {
	// Plane sweep on level=1
	int planeSweepLevel = 0;
	checkCudaErrors(cudaMemset(ps_error, 0, dataSize));
	checkCudaErrors(cudaMemset(ps_depth, 0, dataSize));
	checkCudaErrors(cudaMemset(ps_disparity, 0, dataSize));
	Clone(pI1[planeSweepLevel], pW[planeSweepLevel], pH[planeSweepLevel], pS[planeSweepLevel], ps_i1warp);
	SetValue(ps_error, planeSweepMaxError, pW[planeSweepLevel], pH[planeSweepLevel], pS[planeSweepLevel]);
	for (int sweep = 0; sweep < planeSweepMaxDisparity; sweep += planeSweepStride) {
		PlaneSweepCorrelation(ps_i1warp, pI0[planeSweepLevel], ps_disparity, sweep, planeSweepWindow,
			pW[planeSweepLevel], pH[planeSweepLevel], pS[planeSweepLevel], ps_error);
		for (int psStride = 0; psStride < planeSweepStride; psStride++) {
			WarpImage(ps_i1warp, pW[planeSweepLevel], pH[planeSweepLevel], pS[planeSweepLevel],
				d_tvxBackward, d_tvyBackward, ps_i1warps);
			Swap(ps_i1warp, ps_i1warps);
		}
	}
	return 0;
}

int Stereo::planeSweepOcclusionCheck() {
	isPlaneSweepOcclusionChecked = true;
	// Get wFinal

	return 0;
}

int Stereo::copyStereoToHost(cv::Mat &wCropped) {
	// Convert Disparity to Depth
	if (isOcclusionChecked) {
		ConvertDisparityToDepth(d_wFinal, baseline, focal, width, height, stride, d_depth);
	}
	else {
		ConvertDisparityToDepth(d_w, baseline, focal, width, height, stride, d_depth);
	}
	
	// Remove Padding
	//checkCudaErrors(cudaMemcpy((float *)depth.ptr(), d_w, stride * height * sizeof(float), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy((float *)depth.ptr(), d_depth, stride * height * sizeof(float), cudaMemcpyDeviceToHost));
	cv::Rect roi(0, 0, width, height); // define roi here as x0, y0, width, height
	wCropped = depth(roi);
	return 0;
}

int Stereo::copyPlaneSweepToHost(cv::Mat &ps) {
	// Convert Disparity to Depth
	if (isPlaneSweepOcclusionChecked) {
		ConvertDisparityToDepth(ps_disparityFinal, baseline, focal, width, height, stride, ps_depth);
	}
	else {
		ConvertDisparityToDepth(ps_disparity, baseline, focal, width, height, stride, ps_depth);
	}
	
	// Remove Padding
	//checkCudaErrors(cudaMemcpy((float *)depth.ptr(), d_w, stride * height * sizeof(float), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy((float *)planeSweepDepth.ptr(), ps_depth, stride * height * sizeof(float), cudaMemcpyDeviceToHost));
	cv::Rect roi(0, 0, width, height); // define roi here as x0, y0, width, height
	ps = planeSweepDepth(roi);
	return 0;
}

int Stereo::solveOpticalFlow() {
	// construct pyramid
	for (int level = 1; level < nLevels; level++) {
		Downscale(pI0[level - 1],
			pW[level - 1], pH[level - 1], pS[level - 1],
			pW[level], pH[level], pS[level],
			pI0[level]);

		Downscale(pI1[level - 1],
			pW[level - 1], pH[level - 1], pS[level - 1],
			pW[level], pH[level], pS[level],
			pI1[level]);
	}

	// solve flow
	checkCudaErrors(cudaMemset(d_u, 0, dataSize));
	checkCudaErrors(cudaMemset(d_v, 0, dataSize));

	for (int level = nLevels - 1; level >= 0; level--) {
		for (int warpIter = 0; warpIter < nWarpIters; warpIter++) {
			//std::cout << level << std::endl;
			//initialize zeros
			checkCudaErrors(cudaMemset(d_du, 0, dataSize));
			checkCudaErrors(cudaMemset(d_dv, 0, dataSize));
			checkCudaErrors(cudaMemset(d_dus, 0, dataSize));
			checkCudaErrors(cudaMemset(d_dvs, 0, dataSize));

			checkCudaErrors(cudaMemset(d_dumed, 0, dataSize));
			checkCudaErrors(cudaMemset(d_dvmed, 0, dataSize));
			checkCudaErrors(cudaMemset(d_dumeds, 0, dataSize));
			checkCudaErrors(cudaMemset(d_dvmeds, 0, dataSize));

			checkCudaErrors(cudaMemset(d_pu1, 0, dataSize));
			checkCudaErrors(cudaMemset(d_pu2, 0, dataSize));
			checkCudaErrors(cudaMemset(d_pv1, 0, dataSize));
			checkCudaErrors(cudaMemset(d_pv2, 0, dataSize));

			//warp frame 1
			WarpImage(pI1[level], pW[level], pH[level], pS[level], d_u, d_v, d_i1warp);
			ComputeDerivatives(pI0[level], d_i1warp, pW[level], pH[level], pS[level], d_Ix, d_Iy, d_Iz);

			//inner iteration
			for (int iter = 0; iter < nSolverIters; ++iter)
			{
				SolveDataL1(d_dumed, d_dvmed,
					d_pu1, d_pu2,
					d_pv1, d_pv2,
					d_Ix, d_Iy, d_Iz,
					pW[level], pH[level], pS[level],
					lambda, theta,
					d_dumeds, d_dvmeds); //du1 = duhat output
				Swap(d_dumed, d_dumeds);
				Swap(d_dvmed, d_dvmeds);

				SolveSmoothDualTVGlobal(d_dumed, d_dvmed,
					d_pu1, d_pu2, d_pv1, d_pv2,
					pW[level], pH[level], pS[level],
					tau, theta,
					d_pu1s, d_pu2s, d_pv1s, d_pv2s);
				Swap(d_pu1, d_pu1s);
				Swap(d_pu2, d_pu2s);
				Swap(d_pv1, d_pv1s);
				Swap(d_pv2, d_pv2s);
				//***********************************

				/*MedianFilter(d_dumed, d_dvmed, pW[level], pH[level], pS[level],
					d_dumeds, d_dvmeds, 5);
				Swap(d_dumed, d_dumeds);
				Swap(d_dvmed, d_dvmeds);*/
			}
			// one median filtering
			MedianFilter(d_dumed, d_dvmed, pW[level], pH[level], pS[level],
				d_dumeds, d_dvmeds, 5);
			Swap(d_dumed, d_dumeds);
			Swap(d_dvmed, d_dvmeds);

			// update u, v
			Add(d_u, d_dumed, pH[level] * pS[level], d_u);
			Add(d_v, d_dvmed, pH[level] * pS[level], d_v);
			/*
						MedianFilter(d_u, d_v, pW[level], pH[level], pS[level],
							d_dumeds, d_dvmeds, 5);
						Swap(d_u, d_dumeds);
						Swap(d_v, d_dvmeds);*/
		}

		//upscale
		if (level > 0)
		{
			// scale uv
			//float scale = (float)pW[level + 1] / (float)pW[level];
			float scale = fScale;

			Upscale(d_u, pW[level], pH[level], pS[level],
				pW[level - 1], pH[level - 1], pS[level - 1], scale, d_us);

			//float scaleY = (float)pH[level + 1] / (float)pH[level];

			Upscale(d_v, pW[level], pH[level], pS[level],
				pW[level - 1], pH[level - 1], pS[level - 1], scale, d_vs);

			Swap(d_u, d_us);
			Swap(d_v, d_vs);
		}
	}
	
	if (visualizeResults) {
		FlowToHSV(d_u, d_v, width, height, stride, d_uvrgb, flowScale);
	}
	//FlowToHSV(d_u, d_v, width, height, stride, d_uvrgb, flowScale);
	//SolveSceneFlow(d_u, d_v, d_depth016u, d_depth116u, width, height, stride, d_sceneflow);
	//std::cout << stride << " " << height << " " << height << " " << inputChannels << std::endl;
	return 0;
}


int Stereo::copyOpticalFlowVisToHost(cv::Mat &uvrgbCropped) {
	// Remove Padding
	checkCudaErrors(cudaMemcpy((float3 *)uvrgb.ptr(), d_uvrgb, width * height * sizeof(float) * 3, cudaMemcpyDeviceToHost));
	cv::Rect roi(0, 0, width, height); // define roi here as x0, y0, width, height
	uvrgbCropped = uvrgb(roi);
	return 0;
}

int Stereo::copyOpticalFlowToHost(cv::Mat &u, cv::Mat &v) {
	checkCudaErrors(cudaMemcpy((float *)upad.ptr(), d_u, stride * height * sizeof(float), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy((float *)vpad.ptr(), d_v, stride * height * sizeof(float), cudaMemcpyDeviceToHost));

	// Remove Padding
	cv::Rect roi(0, 0, width, height); // define roi here as x0, y0, width, height
	u = upad(roi);
	v = vpad(roi);

	return 0;
}

// Align up n to the nearest multiple of m
inline int Stereo::iAlignUp(int n)
{
	int m = this->StrideAlignment;
	int mod = n % m;

	if (mod)
		return n + m - mod;
	else
		return n;
}

int Stereo::iDivUp(int n, int m)
{
	return (n + m - 1) / m;
}

// swap two values
template<typename T>
inline void Stereo::Swap(T &a, T &ax)
{
	T t = a;
	a = ax;
	ax = t;
}

//swap four values
template<typename T>
inline void Stereo::Swap(T &a, T &ax, T &b, T &bx)
{
	Swap(a, ax);
	Swap(b, bx);
}

//swap eight values
template<typename T>
inline void Stereo::Swap(T &a, T &ax, T &b, T &bx, T &c, T &cx, T &d, T &dx)
{
	Swap(a, ax);
	Swap(b, bx);
	Swap(c, cx);
	Swap(d, dx);
}

int Stereo::computePyramidLevels(int width, int height, int minWidth, float scale) {
	int nLevels = 1;
	int pHeight = (int)((float)height / scale);
	while (pHeight > minWidth) {
		nLevels++;
		pHeight = (int)((float)pHeight / scale);
	}
	std::cout << "Pyramid Levels: " << nLevels << std::endl;
	return nLevels;
}

int Stereo::initializeColorWheel() {
	checkCudaErrors(cudaMalloc(&d_colorwheel, 55 * 3 * sizeof(float)));
	float colorwheel[165] = { 255, 0, 0,
		255, 17, 0,
		255, 34, 0,
		255, 51, 0,
		255, 68, 0,
		255, 85, 0,
		255, 102, 0,
		255, 119, 0,
		255, 136, 0,
		255, 153, 0,
		255, 170, 0,
		255, 187, 0,
		255, 204, 0,
		255, 221, 0,
		255, 238, 0,
		255, 255, 0,
		213, 255, 0,
		170, 255, 0,
		128, 255, 0,
		85, 255, 0,
		43, 255, 0,
		0, 255, 0,
		0, 255, 63,
		0, 255, 127,
		0, 255, 191,
		0, 255, 255,
		0, 232, 255,
		0, 209, 255,
		0, 186, 255,
		0, 163, 255,
		0, 140, 255,
		0, 116, 255,
		0, 93, 255,
		0, 70, 255,
		0, 47, 255,
		0, 24, 255,
		0, 0, 255,
		19, 0, 255,
		39, 0, 255,
		58, 0, 255,
		78, 0, 255,
		98, 0, 255,
		117, 0, 255,
		137, 0, 255,
		156, 0, 255,
		176, 0, 255,
		196, 0, 255,
		215, 0, 255,
		235, 0, 255,
		255, 0, 255,
		255, 0, 213,
		255, 0, 170,
		255, 0, 128,
		255, 0, 85,
		255, 0, 43 };
	checkCudaErrors(cudaMemcpy(colorwheel, d_colorwheel, 55 * 3 * sizeof(float), cudaMemcpyDeviceToHost));
	return 0;
}