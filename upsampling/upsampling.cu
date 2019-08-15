#include "upsampling.h"

lup::Upsampling::Upsampling() {
	this->BlockWidth = 32;
	this->BlockHeight = 12;
	this->StrideAlignment = 32;
}

lup::Upsampling::Upsampling(int blockWidth, int blockHeight, int strideAlignment) {
	this->BlockWidth = blockWidth;
	this->BlockHeight = blockHeight;
	this->StrideAlignment = strideAlignment;
}

int lup::Upsampling::initialize(int width, int height, int maxIter, float beta, float gamma,
	float alpha0, float alpha1, float timestep_lambda, float lambda_tgvl2, float maxDepth) {
	// Set memory for lidarinput (32fc1), lidarmask(32fc1), image0, image1 (8uc3), depthout (32fc1)
	// flowinput (32fc2), depthinput (32fc1)
	this->width = width;
	this->height = height;
	this->stride = this->iAlignUp(width);

	this->maxIter = maxIter;
	this->beta = beta;
	this->gamma = gamma;
	this->alpha0 = alpha0;
	this->alpha1 = alpha1;
	this->timestep_lambda = timestep_lambda;
	this->lambda_tgvl2 = lambda_tgvl2;
	this->maxDepth = maxDepth;
	this->isRefined = false;

	dataSize8uc3 = stride * height * sizeof(uchar3);
	dataSize32f = stride * height * sizeof(float);
	dataSize32fc2 = stride * height * sizeof(float2);
	dataSize32fc4 = stride * height * sizeof(float4);

	checkCudaErrors(cudaMalloc(&d_gray0, dataSize32f));
	checkCudaErrors(cudaMalloc(&d_lidar, dataSize32f));
	checkCudaErrors(cudaMalloc(&d_sem, dataSize8uc3));
	checkCudaErrors(cudaMalloc(&d_gray0smooth, dataSize32f));
	checkCudaErrors(cudaMalloc(&d_motionStereo, dataSize32f));
	checkCudaErrors(cudaMalloc(&d_grad, dataSize32f));
	checkCudaErrors(cudaMalloc(&d_i0, dataSize8uc3));

	// 3D
	checkCudaErrors(cudaMalloc(&d_X, dataSize32f));
	checkCudaErrors(cudaMalloc(&d_Y, dataSize32f));
	checkCudaErrors(cudaMalloc(&d_Z, dataSize32f));

	// Process variables
	checkCudaErrors(cudaMalloc(&d_depth0, dataSize32f)); // Propagated depth
	checkCudaErrors(cudaMalloc(&d_a, dataSize32f)); // Tensor
	checkCudaErrors(cudaMalloc(&d_b, dataSize32f));
	checkCudaErrors(cudaMalloc(&d_c, dataSize32f));
	checkCudaErrors(cudaMalloc(&d_w, dataSize32f));
	checkCudaErrors(cudaMalloc(&d_etau, dataSize32f));
	checkCudaErrors(cudaMalloc(&d_etav1, dataSize32f));
	checkCudaErrors(cudaMalloc(&d_etav2, dataSize32f));
	checkCudaErrors(cudaMalloc(&d_p, dataSize32fc2));
	checkCudaErrors(cudaMalloc(&d_q, dataSize32fc4));
	checkCudaErrors(cudaMalloc(&d_u, dataSize32f));
	checkCudaErrors(cudaMalloc(&d_uinit, dataSize32f));
	checkCudaErrors(cudaMalloc(&d_uold, dataSize32f));
	checkCudaErrors(cudaMalloc(&d_u_, dataSize32f));
	checkCudaErrors(cudaMalloc(&d_u_s, dataSize32f));
	checkCudaErrors(cudaMalloc(&d_v, dataSize32fc2));
	checkCudaErrors(cudaMalloc(&d_v_, dataSize32fc2));
	checkCudaErrors(cudaMalloc(&d_v_s, dataSize32fc2));
	checkCudaErrors(cudaMalloc(&d_gradv, dataSize32fc4));
	checkCudaErrors(cudaMalloc(&d_dw, dataSize32f));
	checkCudaErrors(cudaMalloc(&d_d, dataSize32f));
	checkCudaErrors(cudaMalloc(&d_Tp, dataSize32fc2));
	checkCudaErrors(cudaMalloc(&d_depth0sub, dataSize32f));
	checkCudaErrors(cudaMalloc(&d_lidarsub, dataSize32f));
	checkCudaErrors(cudaMalloc(&d_depth0norefinement, dataSize32f));

	checkCudaErrors(cudaMalloc(&d_lidarArea, dataSize32f));
	checkCudaErrors(cudaMalloc(&d_lidarAlpha, dataSize32f));

	// Debugging
	checkCudaErrors(cudaMalloc(&debug_depth, dataSize32f));

	// Output
	checkCudaErrors(cudaMalloc(&d_depth, dataSize32f));


	/*checkCudaErrors(cudaMalloc(&d_depth, dataSize32f));
	checkCudaErrors(cudaMalloc(&d_depth0, dataSize32f));
	checkCudaErrors(cudaMalloc(&d_u0, dataSize32f));
	checkCudaErrors(cudaMalloc(&d_v0, dataSize32f));
	checkCudaErrors(cudaMalloc(&d_i0, dataSize8uc3));
	checkCudaErrors(cudaMalloc(&d_i1, dataSize8uc3));
	*/

	return 0;
}

lup::Upsampling::~Upsampling() {
	// Inputs and outputs
	checkCudaErrors(cudaFree(d_gray0));
	checkCudaErrors(cudaFree(d_lidar));
	checkCudaErrors(cudaFree(d_sem));
	checkCudaErrors(cudaFree(d_gray0smooth));
	checkCudaErrors(cudaFree(d_motionStereo));
	checkCudaErrors(cudaFree(d_grad));
	checkCudaErrors(cudaFree(d_i0));

	checkCudaErrors(cudaFree(d_X));
	checkCudaErrors(cudaFree(d_Y));
	checkCudaErrors(cudaFree(d_Z));

	// Process variables
	checkCudaErrors(cudaFree(d_depth0));
	checkCudaErrors(cudaFree(d_a));
	checkCudaErrors(cudaFree(d_b));
	checkCudaErrors(cudaFree(d_c));
	checkCudaErrors(cudaFree(d_w));
	checkCudaErrors(cudaFree(d_etau));
	checkCudaErrors(cudaFree(d_etav1));
	checkCudaErrors(cudaFree(d_etav2));
	checkCudaErrors(cudaFree(d_p));
	checkCudaErrors(cudaFree(d_q));
	checkCudaErrors(cudaFree(d_u));
	checkCudaErrors(cudaFree(d_uinit));
	checkCudaErrors(cudaFree(d_uold));
	checkCudaErrors(cudaFree(d_u_));
	checkCudaErrors(cudaFree(d_u_s));
	checkCudaErrors(cudaFree(d_v));
	checkCudaErrors(cudaFree(d_v_));
	checkCudaErrors(cudaFree(d_v_s));

	checkCudaErrors(cudaFree(d_gradv));
	checkCudaErrors(cudaFree(d_dw));
	checkCudaErrors(cudaFree(d_d));
	checkCudaErrors(cudaFree(d_depth0sub));
	checkCudaErrors(cudaFree(d_lidarsub));
	checkCudaErrors(cudaFree(d_depth0norefinement));
	checkCudaErrors(cudaFree(d_lidarArea));
	checkCudaErrors(cudaFree(d_lidarAlpha));

	checkCudaErrors(cudaFree(d_Tp));

	// Debugging
	checkCudaErrors(cudaFree(debug_depth));

	// Output
	checkCudaErrors(cudaFree(d_depth));
}

int lup::Upsampling::copyImagesToDevice(cv::Mat gray, cv::Mat sparseDepth) {
	if (gray.type() == CV_8U) {
		// Convert to 32F
		cv::Mat trueGray;
		gray.convertTo(trueGray, CV_32F, 1.0 / 256.0);
		checkCudaErrors(cudaMemcpy(d_gray0, (float *)trueGray.ptr(), dataSize32f, cudaMemcpyHostToDevice));
	}
	else {
		checkCudaErrors(cudaMemcpy(d_gray0, (float *)gray.ptr(), dataSize32f, cudaMemcpyHostToDevice));
	}
	checkCudaErrors(cudaMemcpy(d_lidar, (float *)sparseDepth.ptr(), dataSize32f, cudaMemcpyHostToDevice));
	return 0;
}

int lup::Upsampling::copyImagesToDevice(cv::Mat gray, cv::Mat lidar, cv::Mat semantic) {
	// Gaussian filter gray image
	//cv::Mat grayGauss = cv::Mat::zeros(gray.size(), CV_32F);
	//cv::GaussianBlur(gray, grayGauss, cv::Size(3, 3), 0, 0);

	checkCudaErrors(cudaMemcpy(d_gray0, (float *)gray.ptr(), dataSize32f, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_lidar, (float *)lidar.ptr(), dataSize32f, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_sem, (uchar3 *)semantic.ptr(), dataSize8uc3, cudaMemcpyHostToDevice));
	//checkCudaErrors(cudaMemcpy(d_gray0smooth, (float *)grayGauss.ptr(), dataSize32f, cudaMemcpyHostToDevice));
	return 0;
}

int lup::Upsampling::copyImagesToDeviceTGVL2test(cv::Mat gray, cv::Mat lidar, cv::Mat depthInit, cv::Mat semantic) {

	checkCudaErrors(cudaMemcpy(d_gray0, (float *)gray.ptr(), dataSize32f, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_lidar, (float *)lidar.ptr(), dataSize32f, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_sem, (uchar3 *)semantic.ptr(), dataSize8uc3, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_depth0, (float *)depthInit.ptr(), dataSize32f, cudaMemcpyHostToDevice));
	return 0;
}

int lup::Upsampling::copyImagesToDevice(cv::Mat gray, cv::Mat lidar, cv::Mat motionStereo, cv::Mat semantic)
{
	std::cout << "Images copied to GPU." << std::endl;
	checkCudaErrors(cudaMemcpy(d_gray0, (float *)gray.ptr(), dataSize32f, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_lidar, (float *)lidar.ptr(), dataSize32f, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_sem, (uchar3 *)semantic.ptr(), dataSize8uc3, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_motionStereo, (float *)motionStereo.ptr(), dataSize32f, cudaMemcpyHostToDevice));
	return 0;
}

int lup::Upsampling::copyImagesToDevice(cv::Mat rgb, cv::Mat gray, cv::Mat lidar, cv::Mat motionStereo, cv::Mat semantic)
{
	std::cout << "Images copied to GPU." << std::endl;
	checkCudaErrors(cudaMemcpy(d_i0, (uchar3 *)rgb.ptr(), dataSize8uc3, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_gray0, (float *)gray.ptr(), dataSize32f, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_lidar, (float *)lidar.ptr(), dataSize32f, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_sem, (uchar3 *)semantic.ptr(), dataSize8uc3, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_motionStereo, (float *)motionStereo.ptr(), dataSize32f, cudaMemcpyHostToDevice));
	return 0;
}

int lup::Upsampling::propagateColorOnly(int radius) {
	Gradient(d_gray0, d_grad);
	//cudaMemset(d_depth0sub, 0, dataSize32f);
	cudaMemset(d_depth0, 0, dataSize32f);
	PropagateColorOnly(d_grad, d_lidar, d_depth0, radius);
	//Clone(d_depth0sub, d_depth0);
	return 0;
}

int lup::Upsampling::optimizeOnly() {
	//std::cout << "Solving TGVL2..." << std::endl;
	isRefined = true;
	Gradient(d_gray0, d_grad);
	// Calculate weight
	CalcWeight(d_lidar, d_w, lambda_tgvl2);
	//showImage("dgrad", d_w, true);

	// Normalize depth
	//Mult(d_depth0, 1.0f / 40.0f, d_uinit);
	NormalizeClip(d_lidar, 0.0f, maxDepth, d_uinit);
	//showImage("lidar", d_lidar, 0.0f, 1.0f, false);
	Clone(d_lidar, d_uinit); // Copy back to depth0
	//showImage("lidar2", d_lidar, 0.0f, 1.0f, false);
	//showDepthJet("uinit", d_uinit, false);
	//showDepthJet("propagate", d_depth0, false);

	// Solve TGVL2
	//Clone(d_uinit, d_velo);
	//Clone(d_d, d_velo);
	//Clone(d_uinit, d_lidar);
	Clone(d_d, d_lidar);
	upsamplingTensorTGVL2(maxIter, beta, gamma, alpha0, alpha1, timestep_lambda);
	Clone(d_depth, d_u);
	return 0;
}

int lup::Upsampling::solve() {
	//std::cout << "Solving TGVL2..." << std::endl;
	isRefined = true;

	// Calculate weight
	CalcWeight(d_depth0, d_w, lambda_tgvl2);
	//showImage("dgrad", d_w, true);

	// Normalize depth
	//Mult(d_depth0, 1.0f / 40.0f, d_uinit);
	NormalizeClip(d_depth0, 0.0f, maxDepth, d_uinit);
	Clone(d_depth0, d_uinit); // Copy back to depth0
	//showDepthJet("lidar", d_lidar, false);
	//showDepthJet("propagate", d_depth0, false);

	// Solve TGVL2
	//Clone(d_uinit, d_velo);
	//Clone(d_d, d_velo);
	Clone(d_uinit, d_depth0);
	Clone(d_d, d_depth0);
	upsamplingTensorTGVL2(maxIter, beta, gamma, alpha0, alpha1, timestep_lambda);
	Clone(d_depth, d_u);

	return 0;
}

int lup::Upsampling::upsamplingTensorTGVL2(int maxIter, float beta, float gamma,
	float alpha0, float alpha1, float timestep_lambda)
{
	int M = height;
	int N = width;

	// Initial time-steps
	float tau = 1.0f;
	float sigma = 1.0f / tau;

	// Preconditioning variables
	float eta_p = 3.0f;
	float eta_q = 2.0f;

	// Calculate anisotropic diffucion tensor
	Gaussian(d_gray0, d_gray0smooth);
	CalcTensor(d_gray0smooth, beta, gamma, 2, d_a, d_b, d_c);
	//showImage("tensor", d_a, false);

	SolveEta(d_w, alpha0, alpha1, d_a, d_b, d_c, d_etau, d_etav1, d_etav2);
	checkCudaErrors(cudaMemset(d_p, 0, dataSize32fc2));
	checkCudaErrors(cudaMemset(d_q, 0, dataSize32fc4));
	Clone(d_u, d_uinit);
	Clone(d_u_, d_u);
	checkCudaErrors(cudaMemset(d_uold, 0, dataSize32f));
	checkCudaErrors(cudaMemset(d_v, 0, dataSize32fc2));
	Clone(d_v_, d_v);
	checkCudaErrors(cudaMemset(d_gradv, 0, dataSize32fc4));
	Mult(d_d, d_w, d_dw);

	bool firstIter = false;

	for (int iter = 0; iter < maxIter; iter++) {
		float mu;
		if (sigma < 1000) {
			mu = 1.0f / sqrt(1.0f + 0.7f* tau * timestep_lambda);
		}
		else {
			mu = 1.0f;
		}
		UpdateDualVariablesTGV(d_u_, d_v_, alpha0, alpha1, sigma, eta_p, eta_q, d_a, d_b, d_c, d_gradv, d_p, d_q);
		Clone(d_u_, d_u);
		Clone(d_v_, d_v);
		SolveTp(d_a, d_b, d_c, d_p, d_Tp);
		UpdatePrimalVariablesL2(d_Tp, d_u_, d_v_, d_p, d_q, d_a, d_b, d_c,
			tau, d_etau, d_etav1, d_etav2,
			alpha0, alpha1, d_w, d_dw, mu,
			d_u, d_v, d_u_s, d_v_s);
		Clone(d_u_, d_u_s);
		Clone(d_v_, d_v_s);

		sigma = sigma / mu;
		tau = tau * mu;

		//if (iter % 1 == 0) {
		//	cv::Mat u_result = cv::Mat::zeros(cv::Size(stride, height), CV_32F);
		//	checkCudaErrors(cudaMemcpy((float *)u_result.ptr(), d_u, dataSize32f, cudaMemcpyDeviceToHost));
		//	cv::Mat u_color, u_gray, u_norm;
		//	//u_norm = (u_result / 40.0f )*256.0f;
		//	u_norm = u_result*256.0f;
		//	//cv::normalize(u_result, u_norm, 0, 255, cv::NORM_MINMAX);
		//	u_norm.convertTo(u_gray, CV_8UC1);
		//	cv::applyColorMap(u_gray, u_color, cv::COLORMAP_JET);
		//	cv::putText(u_color, std::to_string((iter*100)/maxIter), cvPoint(20, 20),
		//		cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(200, 200, 250), 1, CV_AA);
		//	cv::imshow("u_result", u_color);
		//	if (!firstIter && iter!=0) {
		//		cv::imshow("firstIter", u_color);
		//		firstIter = true;
		//	}
		//	cv::waitKey(1);
		//}
	}

	//std::cout << "Success" << std::endl;
	return 0;
}

int lup::Upsampling::copyImagesToHost(cv::Mat depth) {
	if (isRefined) {
		//std::cout << "Refined (TGVL2-processed) results copied to host." << std::endl;
		checkCudaErrors(cudaMemcpy((float *)depth.ptr(), d_depth, dataSize32f, cudaMemcpyDeviceToHost));
	}
	else {
		//std::cout << "Propagation (no refinement with TGVL2) results copied to host." << std::endl;
		//float* input_sub;
		//checkCudaErrors(cudaMalloc(&input_sub, dataSize32f));
		NormalizeClip(d_depth0, 0.0f, maxDepth, d_depth0norefinement);
		checkCudaErrors(cudaMemcpy((float *)depth.ptr(), d_depth0norefinement, dataSize32f, cudaMemcpyDeviceToHost));
	}

	return 0;
}

int lup::Upsampling::copyImagesToHost(cv::Mat depth, cv::Mat propagated) {
	if (isRefined) {
		std::cout << "Refined (TGVL2-processed) results copied to host." << std::endl;
		checkCudaErrors(cudaMemcpy((float *)depth.ptr(), d_depth, dataSize32f, cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy((float *)propagated.ptr(), d_depth0, dataSize32f, cudaMemcpyDeviceToHost));
	}
	else {
		std::cout << "Propagation (no refinement with TGVL2) results copied to host." << std::endl;
		//float* input_sub;
		//checkCudaErrors(cudaMalloc(&input_sub, dataSize32f));
		NormalizeClip(d_depth0, 0.0f, maxDepth, d_depth0norefinement);
		checkCudaErrors(cudaMemcpy((float *)depth.ptr(), d_depth0norefinement, dataSize32f, cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy((float *)propagated.ptr(), d_depth0norefinement, dataSize32f, cudaMemcpyDeviceToHost));
	}

	return 0;
}

int lup::Upsampling::convertDepthTo3D(float focal, float cx, float cy) {
	ConvertDepthTo3D(d_depth, d_X, d_Y, d_Z, focal, cx, cy);
	return 0;
}

int lup::Upsampling::copy3dToHost(cv::Mat X, cv::Mat Y, cv::Mat Z) {
	checkCudaErrors(cudaMemcpy((float *)X.ptr(), d_X, stride * height * sizeof(float), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy((float *)Y.ptr(), d_Y, stride * height * sizeof(float), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy((float *)Z.ptr(), d_Z, stride * height * sizeof(float), cudaMemcpyDeviceToHost));
	return 0;
}

// DEBUGGING
int lup::Upsampling::showImage(std::string windowName, float* input, bool shouldWait) {
	cv::Mat image = cv::Mat::zeros(cv::Size(stride, height), CV_32F);
	checkCudaErrors(cudaMemcpy((float *)image.ptr(), input, dataSize32f, cudaMemcpyDeviceToHost));
	cv::imshow(windowName, image);
	if (shouldWait) cv::waitKey();
}

int lup::Upsampling::showImage(std::string windowName, float* input, float minVal, float maxVal, bool shouldWait) {
	cv::Mat image = cv::Mat::zeros(cv::Size(stride, height), CV_32F);
	float* input_sub;
	checkCudaErrors(cudaMalloc(&input_sub, dataSize32f));
	NormalizeClip(input, minVal, maxVal, input_sub);
	checkCudaErrors(cudaMemcpy((float *)image.ptr(), input_sub, dataSize32f, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaFree(input_sub));

	cv::Mat u_norm, u_gray, u_color;
	u_norm = image * 256.0f;
	u_norm.convertTo(u_gray, CV_8UC1);
	cv::applyColorMap(u_gray, u_color, cv::COLORMAP_JET);

	cv::imshow(windowName, u_color);
	if (shouldWait) cv::waitKey();
}

int lup::Upsampling::showImage(std::string windowName, uchar3* input, bool shouldWait) {
	cv::Mat image = cv::Mat::zeros(cv::Size(stride, height), CV_8UC3);
	checkCudaErrors(cudaMemcpy((uchar3 *)image.ptr(), input, dataSize8uc3, cudaMemcpyDeviceToHost));
	cv::imshow(windowName, image);
	if (shouldWait) cv::waitKey();
}

int lup::Upsampling::saveDepthJet(std::string filename, float* input, float maxDepth)
{
	cv::Mat image = cv::Mat::zeros(cv::Size(stride, height), CV_32F);
	float* input_sub;
	checkCudaErrors(cudaMalloc(&input_sub, dataSize32f));
	NormalizeClip(input, 0.0f, maxDepth, input_sub);
	checkCudaErrors(cudaMemcpy((float *)image.ptr(), input_sub, dataSize32f, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaFree(input_sub));

	cv::Mat u_norm, u_gray, u_color;
	u_norm = image * 256.0f;
	u_norm.convertTo(u_gray, CV_8UC1);
	cv::applyColorMap(u_gray, u_color, cv::COLORMAP_JET);

	cv::imwrite(filename, u_color);

	return 0;
}

int lup::Upsampling::showDepthJet(std::string windowName, float* input, bool shouldWait) {
	cv::Mat image = cv::Mat::zeros(cv::Size(stride, height), CV_32F);
	float* input_sub;
	checkCudaErrors(cudaMalloc(&input_sub, dataSize32f));
	NormalizeClip(input, 0.0f, maxDepth, input_sub);
	checkCudaErrors(cudaMemcpy((float *)image.ptr(), input_sub, dataSize32f, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaFree(input_sub));

	cv::Mat u_norm, u_gray, u_color;
	u_norm = image * 256.0f;
	u_norm.convertTo(u_gray, CV_8UC1);
	cv::applyColorMap(u_gray, u_color, cv::COLORMAP_JET);

	cv::imshow(windowName, u_color);
	if (shouldWait) cv::waitKey();
}

void  lup::Upsampling::showDepthJet(std::string windowName, float* input, float maxDepth, bool shouldWait = true) {
	cv::Mat image = cv::Mat::zeros(cv::Size(stride, height), CV_32F);
	float* input_sub;
	checkCudaErrors(cudaMalloc(&input_sub, dataSize32f));
	NormalizeClip(input, 0.0f, maxDepth, input_sub);
	checkCudaErrors(cudaMemcpy((float *)image.ptr(), input_sub, dataSize32f, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaFree(input_sub));

	cv::Mat u_norm, u_gray, u_color;
	u_norm = image * 256.0f;
	u_norm.convertTo(u_gray, CV_8UC1);
	cv::applyColorMap(u_gray, u_color, cv::COLORMAP_JET);

	cv::imshow(windowName, u_color);
	if (shouldWait) cv::waitKey();
}



// UTILITIES
int lup::Upsampling::iAlignUp(int n)
{
	int m = this->StrideAlignment;
	int mod = n % m;

	if (mod)
		return n + m - mod;
	else
		return n;
}

int lup::Upsampling::iDivUp(int n, int m)
{
	return (n + m - 1) / m;
}

template<typename T> void lup::Upsampling::Swap(T &a, T &ax)
{
	T t = a;
	a = ax;
	ax = t;
}

template<typename T> void lup::Upsampling::Copy(T &dst, T &src)
{
	dst = src;
}
