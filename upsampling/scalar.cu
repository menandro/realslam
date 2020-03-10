#include "stereotgv.h"

__global__
void ScalarMultiplyKernel(float* src, float scalar,
	int width, int height, int stride)
{
	const int ix = threadIdx.x + blockIdx.x * blockDim.x;
	const int iy = threadIdx.y + blockIdx.y * blockDim.y;

	const int pos = ix + iy * stride;

	if (ix >= width || iy >= height) return;

	src[pos] = src[pos] * scalar;
}

void StereoTgv::ScalarMultiply(float *src, float scalar, int w, int h, int s)
{
	dim3 threads(BlockWidth, BlockHeight);
	dim3 blocks(iDivUp(w, threads.x), iDivUp(h, threads.y));

	ScalarMultiplyKernel << <blocks, threads >> > (src, scalar, w, h, s);
}

__global__
void ScalarMultiplyKernel(float* src, float scalar,
	int width, int height, int stride, float* dst)
{
	const int ix = threadIdx.x + blockIdx.x * blockDim.x;
	const int iy = threadIdx.y + blockIdx.y * blockDim.y;

	const int pos = ix + iy * stride;

	if (ix >= width || iy >= height) return;

	dst[pos] = src[pos] * scalar;
}

void StereoTgv::ScalarMultiply(float *src, float scalar, int w, int h, int s, float* dst)
{
	dim3 threads(BlockWidth, BlockHeight);
	dim3 blocks(iDivUp(w, threads.x), iDivUp(h, threads.y));

	ScalarMultiplyKernel << <blocks, threads >> > (src, scalar, w, h, s, dst);
}