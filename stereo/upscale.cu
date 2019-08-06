#include "stereo.h"

/// scalar field to upscale
texture<float, 2, cudaReadModeElementType> texCoarse;

__global__ void UpscaleKernel(int width, int height, int stride, float scale, float *out)
{
	const int ix = threadIdx.x + blockIdx.x * blockDim.x;
	const int iy = threadIdx.y + blockIdx.y * blockDim.y;

	if (ix >= width || iy >= height) return;

	float x = ((float)ix + 0.5f) / (float)width;
	float y = ((float)iy + 0.5f) / (float)height;

	// exploit hardware interpolation
	// and scale interpolated vector to match next pyramid level resolution
	out[ix + iy * stride] = tex2D(texCoarse, x, y) * scale;
}

void Stereo::Upscale(const float *src, int width, int height, int stride,
	int newWidth, int newHeight, int newStride, float scale, float *out)
{
	dim3 threads(BlockWidth, BlockHeight);
	dim3 blocks(iDivUp(newWidth, threads.x), iDivUp(newHeight, threads.y));

	// mirror if a coordinate value is out-of-range
	texCoarse.addressMode[0] = cudaAddressModeMirror;
	texCoarse.addressMode[1] = cudaAddressModeMirror;
	texCoarse.filterMode = cudaFilterModeLinear;
	texCoarse.normalized = true;

	cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();

	cudaBindTexture2D(0, texCoarse, src, width, height, stride * sizeof(float));

	UpscaleKernel <<< blocks, threads >>> (newWidth, newHeight, newStride, scale, out);
}