#include "stereo.h"

__global__
void CloneKernel(const float *src, int width, int height, int stride, float *dst)
{
	const int ix = threadIdx.x + blockIdx.x * blockDim.x;
	const int iy = threadIdx.y + blockIdx.y * blockDim.y;

	const int pos = ix + iy * stride;

	if (ix >= width || iy >= height) return;

	dst[pos] = src[pos];
}

void Stereo::Clone(const float *src, int w, int h, int s, float *dst)
{
	dim3 threads(BlockWidth, BlockHeight);
	dim3 blocks(iDivUp(w, threads.x), iDivUp(h, threads.y));
	CloneKernel << < blocks, threads >> > (src, w, h, s, dst);
}

// Set Value
__global__
void SetValueKernel(float *image, float value, int width, int height, int stride)
{
	const int ix = threadIdx.x + blockIdx.x * blockDim.x;
	const int iy = threadIdx.y + blockIdx.y * blockDim.y;

	const int pos = ix + iy * stride;

	if (ix >= width || iy >= height) return;

	image[pos] = value;
}

void Stereo::SetValue(float *image, float value, int w, int h, int s)
{
	dim3 threads(BlockWidth, BlockHeight);
	dim3 blocks(iDivUp(w, threads.x), iDivUp(h, threads.y));
	SetValueKernel << < blocks, threads >> > (image, value, w, h, s);
}
