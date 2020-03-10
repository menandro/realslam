#include "stereolite.h"

// ***********************
// MULTIPLY
// ***********************
__global__
void LiteScalarMultiplyKernel(float* src, float scalar,
	int width, int height, int stride)
{
	const int ix = threadIdx.x + blockIdx.x * blockDim.x;
	const int iy = threadIdx.y + blockIdx.y * blockDim.y;

	const int pos = ix + iy * stride;

	if (ix >= width || iy >= height) return;

	src[pos] = src[pos] * scalar;
}

void StereoLite::ScalarMultiply(float *src, float scalar, int w, int h, int s)
{
	dim3 threads(BlockWidth, BlockHeight);
	dim3 blocks(iDivUp(w, threads.x), iDivUp(h, threads.y));

	LiteScalarMultiplyKernel << <blocks, threads >> > (src, scalar, w, h, s);
}

__global__
void LiteScalarMultiplyKernel(float* src, float scalar,
	int width, int height, int stride, float* dst)
{
	const int ix = threadIdx.x + blockIdx.x * blockDim.x;
	const int iy = threadIdx.y + blockIdx.y * blockDim.y;

	const int pos = ix + iy * stride;

	if (ix >= width || iy >= height) return;

	dst[pos] = src[pos] * scalar;
}

void StereoLite::ScalarMultiply(float *src, float scalar, int w, int h, int s, float* dst)
{
	dim3 threads(BlockWidth, BlockHeight);
	dim3 blocks(iDivUp(w, threads.x), iDivUp(h, threads.y));

	LiteScalarMultiplyKernel << <blocks, threads >> > (src, scalar, w, h, s, dst);
}

__global__
void LiteScalarMultiplyKernel(float2* src, float scalar,
	int width, int height, int stride, float2* dst)
{
	const int ix = threadIdx.x + blockIdx.x * blockDim.x;
	const int iy = threadIdx.y + blockIdx.y * blockDim.y;

	const int pos = ix + iy * stride;

	if (ix >= width || iy >= height) return;

	float2 srcs = src[pos];
	dst[pos].x = srcs.x * scalar;
	dst[pos].y = srcs.y * scalar;
}

void StereoLite::ScalarMultiply(float2 *src, float scalar, int w, int h, int s, float2* dst)
{
	dim3 threads(BlockWidth, BlockHeight);
	dim3 blocks(iDivUp(w, threads.x), iDivUp(h, threads.y));

	LiteScalarMultiplyKernel << <blocks, threads >> > (src, scalar, w, h, s, dst);
}

//************************
// ADD
//************************
__global__
void LiteAddKernel(float* src1, float * src2,
	int width, int height, int stride, float* dst)
{
	const int ix = threadIdx.x + blockIdx.x * blockDim.x;
	const int iy = threadIdx.y + blockIdx.y * blockDim.y;

	const int pos = ix + iy * stride;

	if (ix >= width || iy >= height) return;

	dst[pos] = src1[pos] + src2[pos];
}

void StereoLite::Add(float *src1, float* src2, int w, int h, int s, float* dst)
{
	dim3 threads(BlockWidth, BlockHeight);
	dim3 blocks(iDivUp(w, threads.x), iDivUp(h, threads.y));

	LiteAddKernel << <blocks, threads >> > (src1, src2, w, h, s, dst);
}

//************************
// ADD FLOAT2
//************************
__global__
void LiteAddFloat2Kernel(float2* src1, float2 * src2,
	int width, int height, int stride, float2* dst)
{
	const int ix = threadIdx.x + blockIdx.x * blockDim.x;
	const int iy = threadIdx.y + blockIdx.y * blockDim.y;

	const int pos = ix + iy * stride;

	if (ix >= width || iy >= height) return;

	dst[pos].x = src1[pos].x + src2[pos].x;
	dst[pos].y = src1[pos].y + src2[pos].y;
}

void StereoLite::Add(float2 *src1, float2* src2, int w, int h, int s, float2* dst)
{
	dim3 threads(BlockWidth, BlockHeight);
	dim3 blocks(iDivUp(w, threads.x), iDivUp(h, threads.y));

	LiteAddFloat2Kernel << <blocks, threads >> > (src1, src2, w, h, s, dst);
}


//************************
// SUBTRACT
//************************
__global__
void LiteSubtractKernel(float* minuend, float * subtrahend,
	int width, int height, int stride, float* difference)
{
	const int ix = threadIdx.x + blockIdx.x * blockDim.x;
	const int iy = threadIdx.y + blockIdx.y * blockDim.y;

	const int pos = ix + iy * stride;

	if (ix >= width || iy >= height) return;

	difference[pos] = minuend[pos] - subtrahend[pos];
}

void StereoLite::Subtract(float *minuend, float* subtrahend, int w, int h, int s, float* difference)
{
	dim3 threads(BlockWidth, BlockHeight);
	dim3 blocks(iDivUp(w, threads.x), iDivUp(h, threads.y));

	LiteSubtractKernel << <blocks, threads >> > (minuend, subtrahend, w, h, s, difference);
}