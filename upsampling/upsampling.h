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

#define CUDA_LIB_PATH "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.1/lib/x64/"
#pragma comment(lib, CUDA_LIB_PATH "cudart.lib")

//#include "common.h"

namespace lup {
	class Upsampling {
	public:
		Upsampling();
		Upsampling(int blockWidth, int blockHeight, int strideAlignment);
		~Upsampling();

		int BlockWidth, BlockHeight, StrideAlignment;

		int width;
		int height;
		int stride;
		int dataSize8uc3;
		int dataSize32f;
		int dataSize32fc2;
		int dataSize32fc4;

		// Inputs and outputs
		float* d_depth; // Output depth
		float* d_depth0; // propagated depth
		uchar3* d_i0; // Image 0
		//uchar3* d_i1; // Image 1
		float* d_gray0; // Reference gray image
		//float* d_gray1; // Next frame gray image
		float* d_gray0smooth; // Gaussian filtered (3x3) gray image
		uchar3* d_sem; // Input semantic image
		float* d_lidar; // Sparse velodyne depth
		float* d_motionStereo; // Depth map from motion stereo
		float* d_grad;
		float* d_X;
		float* d_Y;
		float* d_Z;

		// Process variables
		float *d_a;
		float *d_b;
		float *d_c;
		float *d_etau;
		float *d_etav1;
		float *d_etav2;
		float2 *d_p;
		float4 *d_q;
		float* d_uinit;
		float *d_u;
		float *d_u_;
		float* d_u_s;
		float *d_uold;
		float2 *d_v;
		float2 *d_v_;
		float2 *d_v_s;
		float *d_dw;
		float *d_w; // weights
		float4 *d_gradv;
		float *d_d; // sparse depth
		float2 *d_Tp;
		float* d_depth0sub;
		float* d_lidarsub;
		float* d_depth0norefinement;

		float* d_lidarArea;
		float* d_lidarAlpha;

		// Superpixel
		uchar3* d_superpixel;

		// Debugging
		float* debug_depth;

		// TVGL2
		int maxIter;
		float beta;
		float gamma;
		float alpha0;
		float alpha1;
		float timestep_lambda;
		float lambda_tgvl2;
		float maxDepth;
		bool isRefined;

		// FUNCTIONS
		int initialize(int width, int height, int maxIter, float beta, float gamma,
			float alpha0, float alpha1, float timestep_lambda, float lambda_tgvl2, float maxDepth);

		int copyImagesToDevice(cv::Mat gray, cv::Mat sparseDepth);
		int copyImagesToDevice(cv::Mat gray, cv::Mat lidar, cv::Mat semantic);
		int copyImagesToDeviceTGVL2test(cv::Mat gray, cv::Mat lidar, cv::Mat depthIinit, cv::Mat semantic);
		int copyImagesToDevice(cv::Mat gray, cv::Mat lidar, cv::Mat motionStereo, cv::Mat semantic);
		int copyImagesToDevice(cv::Mat rgb, cv::Mat gray, cv::Mat lidar, cv::Mat motionStereo, cv::Mat semantic);

		int upsamplingTensorTGVL2(int maxIter, float beta, float gamma,
			float alpha0, float alpha1, float timestep_lambda);
		int copyImagesToHost(cv::Mat depth);
		int copyImagesToHost(cv::Mat depth, cv::Mat propagated);
		int copy3dToHost(cv::Mat X, cv::Mat Y, cv::Mat Z);

		int propagateColorOnly(int radius);
		int solve();
		int optimizeOnly();
		int convertDepthTo3D(float focal, float cx, float cy); // convert depth to 3D points

		// UTILITIES
		int iAlignUp(int n);
		int iDivUp(int n, int m);
		template<typename T> void Swap(T &a, T &ax);
		template<typename T> void Copy(T &dst, T &src);


		// CUDA KERNELS
		void CalcWeight(float *input, float *weight, float lambda_tgvl2);
		void CalcTensor(float* gray, float beta, float gamma, int size_grad,
			float* atensor, float* btensor, float* ctensor);
		void SolveEta(float* weights, float alpha0, float alpha1,
			float* a, float *b, float* c,
			float* etau, float* etav1, float* etav2);
		void Mult(float* input0, float* input1, float* output);
		void Mult(float* input0, float scale, float* output);
		void SolveTp(float* a, float* b, float* c, float2* p, float2* Tp);
		void Clone(float* dst, float* src);
		void Clone(float2* dst, float2* src);
		void NormalizeClip(float* input, float min, float max, float *output);
		void DenormalizeClip(float* input, float min, float max, float *output);
		void Normalize(float* input, float min, float max, float *output);
		void Gaussian(float* input, float* output);

		void UpdateDualVariablesTGV(float* u_, float2 *v_, float alpha0, float alpha1, float sigma, float eta_p, float eta_q,
			float* a, float* b, float* c,
			float4* grad_v, float2* p, float4* q);
		void UpdatePrimalVariablesL2(float2* Tp, float* u_, float2* v_, float2* p, float4* q,
			float* a, float* b, float* c,
			float tau, float* eta_u, float* eta_v1, float* eta_v2,
			float alpha0, float alpha1, float* w, float* dw, float mu,
			float* u, float2* v, float* u_s, float2* v_s);
		void PropagateColorOnly(float* grad, float* lidar, float* depthOut, int radius);
		void Gradient(float* input, float* output);
		void PropagateNearestNeighbor(float* im, float* lidar, uchar3* semantic, float* motionStereo, float* depthOut);

		void ConvertDepthTo3D(float* depth, float* X, float* Y, float* Z, float focal);
		void ConvertDepthTo3D(float* depth, float* X, float* Y, float* Z, float focal, float cx, float cy);
		void ConvertDepthTo3D(float* depth, float* X, float* Y, float* Z, uchar3* sem, float focal);
		void ConvertDepthTo3D(float* depth, float* X, float* Y, float* Z, uchar3* sem, float focal, float cx, float cy);
		void ConvertDepthTo3D(float* depth, float* X, float* Y, float* Z, float* grad, float focal);

		// Debugging
		int showImage(std::string windowName, float* input, bool shouldWait);
		int showImage(std::string windowName, float* input, float minVal, float maxVal, bool shouldWait);
		int showImage(std::string windowName, uchar3* input, bool shouldWait);
		int showDepthJet(std::string windowName, float* input, bool shouldWait);
		void showDepthJet(std::string windowName, float* input, float maxDepth, bool shouldWait);
		int saveDepthJet(std::string filename, float* input, float maxDepth);
	};
}
