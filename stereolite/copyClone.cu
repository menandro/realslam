#include "stereoLite.h"

__global__ void LiteCloneKernel(float* src, float* dst, int width, int height, int stride) {
	int iy = blockIdx.y * blockDim.y + threadIdx.y;        // current row 
	int ix = blockIdx.x * blockDim.x + threadIdx.x;        // current column 

	if ((iy < height) && (ix < width))
	{
		int pos = ix + iy * stride;
		dst[pos] = src[pos];
	}
}

void StereoLite::Clone(float* src, float* dst, int w, int h, int s) {
	dim3 threads(BlockWidth, BlockHeight);
	dim3 blocks(iDivUp(w, threads.x), iDivUp(h, threads.y));
	LiteCloneKernel << < blocks, threads >> > (src, dst, w, h, s);
}


__global__ void LiteCloneKernel2(float2* src, float2* dst, int width, int height, int stride) {
	int iy = blockIdx.y * blockDim.y + threadIdx.y;        // current row 
	int ix = blockIdx.x * blockDim.x + threadIdx.x;        // current column 

	if ((iy < height) && (ix < width))
	{
		int pos = ix + iy * stride;
		dst[pos] = src[pos];
	}
}

void StereoLite::Clone(float2* src, float2* dst, int w, int h, int s) {
	dim3 threads(BlockWidth, BlockHeight);
	dim3 blocks(iDivUp(w, threads.x), iDivUp(h, threads.y));
	LiteCloneKernel2 << < blocks, threads >> > (src, dst, w, h, s);
}

// Set Value
__global__
void LiteSetValueKernel(float *image, float value, int width, int height, int stride)
{
	const int ix = threadIdx.x + blockIdx.x * blockDim.x;
	const int iy = threadIdx.y + blockIdx.y * blockDim.y;

	const int pos = ix + iy * stride;

	if (ix >= width || iy >= height) return;

	image[pos] = value;
}

void StereoLite::SetValue(float *image, float value, int w, int h, int s)
{
	dim3 threads(BlockWidth, BlockHeight);
	dim3 blocks(iDivUp(w, threads.x), iDivUp(h, threads.y));
	LiteSetValueKernel << < blocks, threads >> > (image, value, w, h, s);
}