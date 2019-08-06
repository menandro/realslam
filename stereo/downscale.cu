#include "stereo.h"

/// image to downscale
texture<float, 2, cudaReadModeElementType> texFine;

__global__ void DownscaleKernel(int width, int height, int stride, float *out)
{
	const int ix = threadIdx.x + blockIdx.x * blockDim.x;
	const int iy = threadIdx.y + blockIdx.y * blockDim.y;

	if (ix >= width || iy >= height)
	{
		return;
	}

	float dx = 1.0f / (float)width;
	float dy = 1.0f / (float)height;

	float x = ((float)ix + 0.5f) * dx;
	float y = ((float)iy + 0.5f) * dy;

	int pos = ix + iy * stride;

	out[pos] = 0.25f * (tex2D(texFine, x - dx * 0.25f, y) + tex2D(texFine, x + dx * 0.25f, y) +
		tex2D(texFine, x, y - dy * 0.25f) + tex2D(texFine, x, y + dy * 0.25f));
}

__global__ void DownscaleScalingKernel(int width, int height, int stride, float scale, float *out)
{
	const int ix = threadIdx.x + blockIdx.x * blockDim.x;
	const int iy = threadIdx.y + blockIdx.y * blockDim.y;

	if (ix >= width || iy >= height)
	{
		return;
	}

	float dx = 1.0f / (float)width;
	float dy = 1.0f / (float)height;

	float x = ((float)ix + 0.5f) * dx;
	float y = ((float)iy + 0.5f) * dy;

	int pos = ix + iy * stride;

	out[pos] = scale * 0.25f * (tex2D(texFine, x - dx * 0.25f, y) + tex2D(texFine, x + dx * 0.25f, y) +
		tex2D(texFine, x, y - dy * 0.25f) + tex2D(texFine, x, y + dy * 0.25f));
}

void Stereo::Downscale(const float *src, int width, int height, int stride,
	int newWidth, int newHeight, int newStride, float *out)
{
	dim3 threads(BlockWidth, BlockHeight);
	dim3 blocks(iDivUp(newWidth, threads.x), iDivUp(newHeight, threads.y));

	// mirror if a coordinate value is out-of-range
	texFine.addressMode[0] = cudaAddressModeMirror;
	texFine.addressMode[1] = cudaAddressModeMirror;
	texFine.filterMode = cudaFilterModeLinear;
	texFine.normalized = true;

	cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();

	checkCudaErrors(cudaBindTexture2D(0, texFine, src, width, height, stride * sizeof(float)));

	DownscaleKernel << < blocks, threads >> > (newWidth, newHeight, newStride, out);
}

void Stereo::Downscale(const float *src, int width, int height, int stride,
	int newWidth, int newHeight, int newStride, float scale, float *out)
{
	dim3 threads(BlockWidth, BlockHeight);
	dim3 blocks(iDivUp(newWidth, threads.x), iDivUp(newHeight, threads.y));

	// mirror if a coordinate value is out-of-range
	texFine.addressMode[0] = cudaAddressModeMirror;
	texFine.addressMode[1] = cudaAddressModeMirror;
	texFine.filterMode = cudaFilterModeLinear;
	texFine.normalized = true;

	cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();

	checkCudaErrors(cudaBindTexture2D(0, texFine, src, width, height, stride * sizeof(float)));

	DownscaleScalingKernel << < blocks, threads >> > (newWidth, newHeight, newStride, scale, out);
}