#include "stereotgv.h"

__global__ void TgvConvertDisparityToDepthKernel(float *disparity, float baseline,
	float focal, int width, int height, int stride, float *depth)
{
	const int ix = threadIdx.x + blockIdx.x * blockDim.x;
	const int iy = threadIdx.y + blockIdx.y * blockDim.y;

	const int pos = ix + iy * stride;

	if (ix >= width || iy >= height) return;

	depth[pos] = baseline * focal / disparity[pos];
}


void StereoTgv::ConvertDisparityToDepth(float *disparity, float baseline, float focal, int w, int h, int s, float *depth)
{
	dim3 threads(BlockWidth, BlockHeight);
	dim3 blocks(iDivUp(w, threads.x), iDivUp(h, threads.y));

	TgvConvertDisparityToDepthKernel << <blocks, threads >> > (disparity, baseline, focal, w, h, s, depth);
}