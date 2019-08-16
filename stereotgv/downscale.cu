#include "stereotgv.h"

/// image to downscale
texture<float, 2, cudaReadModeElementType> texFine;
texture<float2, 2, cudaReadModeElementType> texFineFloat2;

__global__ void TgvDownscaleKernel(int width, int height, int stride, float *out)
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

void StereoTgv::Downscale(const float *src, int width, int height, int stride,
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

	TgvDownscaleKernel << < blocks, threads >> > (newWidth, newHeight, newStride, out);
}


// *********************************
// Downscaling for Float2
// *********************************
__global__ void TgvDownscaleKernel(int width, int height, int stride, float2 *out)
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

	float2 val00 = tex2D(texFineFloat2, x - dx * 0.25f, y);
	float2 val01 = tex2D(texFineFloat2, x + dx * 0.25f, y);
	float2 val10 = tex2D(texFineFloat2, x, y - dy * 0.25f);
	float2 val11 = tex2D(texFineFloat2, x, y + dy * 0.25f);
	out[pos].x = 0.25f * (val00.x + val01.x + val10.x + val11.x);
	out[pos].y = 0.25f * (val00.y + val01.y + val10.y + val11.y);
}

void StereoTgv::Downscale(const float2 *src, int width, int height, int stride,
	int newWidth, int newHeight, int newStride, float2 *out)
{
	dim3 threads(BlockWidth, BlockHeight);
	dim3 blocks(iDivUp(newWidth, threads.x), iDivUp(newHeight, threads.y));

	// mirror if a coordinate value is out-of-range
	texFineFloat2.addressMode[0] = cudaAddressModeMirror;
	texFineFloat2.addressMode[1] = cudaAddressModeMirror;
	texFineFloat2.filterMode = cudaFilterModeLinear;
	texFineFloat2.normalized = true;

	cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();

	checkCudaErrors(cudaBindTexture2D(0, texFineFloat2, src, width, height, stride * sizeof(float2)));

	TgvDownscaleKernel << < blocks, threads >> > (newWidth, newHeight, newStride, out);
}


// ***********************************
// Downscale with vector downscaling
//************************************

__global__ void TgvDownscaleScalingKernel(int width, int height, int stride, float scale, float *out)
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

void StereoTgv::Downscale(const float *src, int width, int height, int stride,
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

	TgvDownscaleScalingKernel << < blocks, threads >> > (newWidth, newHeight, newStride, scale, out);
}


// ***********************************
// Downscale with vector downscaling for Float2
//************************************

__global__ void TgvDownscaleScalingKernel(int width, int height, int stride, float scale, float2 *out)
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

	float2 val00 = tex2D(texFineFloat2, x - dx * 0.25f, y);
	float2 val01 = tex2D(texFineFloat2, x + dx * 0.25f, y);
	float2 val10 = tex2D(texFineFloat2, x, y - dy * 0.25f);
	float2 val11 = tex2D(texFineFloat2, x, y + dy * 0.25f);
	out[pos].x = scale * 0.25f * (val00.x + val01.x + val10.x + val11.x);
	out[pos].y = scale * 0.25f * (val00.y + val01.y + val10.y + val11.y);
}

void StereoTgv::Downscale(const float2 *src, int width, int height, int stride,
	int newWidth, int newHeight, int newStride, float scale, float2 *out)
{
	dim3 threads(BlockWidth, BlockHeight);
	dim3 blocks(iDivUp(newWidth, threads.x), iDivUp(newHeight, threads.y));

	// mirror if a coordinate value is out-of-range
	texFineFloat2.addressMode[0] = cudaAddressModeMirror;
	texFineFloat2.addressMode[1] = cudaAddressModeMirror;
	texFineFloat2.filterMode = cudaFilterModeLinear;
	texFineFloat2.normalized = true;

	cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();

	checkCudaErrors(cudaBindTexture2D(0, texFineFloat2, src, width, height, stride * sizeof(float)));

	TgvDownscaleScalingKernel << < blocks, threads >> > (newWidth, newHeight, newStride, scale, out);
}

