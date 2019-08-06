#include "stereo.h"

/// image to warp
texture<float, 2, cudaReadModeElementType> texToWarp;

__global__ void WarpingKernel(int width, int height, int stride,
	const float *u, const float *v, float *out)
{
	const int ix = threadIdx.x + blockIdx.x * blockDim.x;
	const int iy = threadIdx.y + blockIdx.y * blockDim.y;

	const int pos = ix + iy * stride;

	if (ix >= width || iy >= height) return;

	float x = ((float)ix + u[pos] + 0.5f) / (float)width;
	float y = ((float)iy + v[pos] + 0.5f) / (float)height;

	out[pos] = tex2D(texToWarp, x, y);
}

void Stereo::WarpImage(const float *src, int w, int h, int s,
	const float *u, const float *v, float *out)
{
	dim3 threads(BlockWidth, BlockHeight);
	dim3 blocks(iDivUp(w, threads.x), iDivUp(h, threads.y));

	// mirror if a coordinate value is out-of-range
	texToWarp.addressMode[0] = cudaAddressModeMirror;
	texToWarp.addressMode[1] = cudaAddressModeMirror;
	texToWarp.filterMode = cudaFilterModeLinear;
	texToWarp.normalized = true;

	cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();

	cudaBindTexture2D(0, texToWarp, src, w, h, s * sizeof(float));

	WarpingKernel << <blocks, threads >> > (w, h, s, u, v, out);
}
