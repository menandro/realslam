#include "stereo.h"

/// image to warp
texture<float, 2, cudaReadModeElementType> texToWarp;
texture<float, 2, cudaReadModeElementType> texTvx;
texture<float, 2, cudaReadModeElementType> texTvy;

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


// **************************************************
// ** Find Warping vector direction (tvx2, tvy2) for Fisheye Stereo
// **************************************************

__global__ void FindWarpingVectorKernel(const float *u, const float * v,
	int width, int height, int stride,
	float *tvx2, float *tvy2)
{
	const int ix = threadIdx.x + blockIdx.x * blockDim.x;
	const int iy = threadIdx.y + blockIdx.y * blockDim.y;

	const int pos = ix + iy * stride;

	if (ix >= width || iy >= height) return;

	float x = ((float)ix + u[pos] + 0.5f) / (float)width;
	float y = ((float)iy + v[pos] + 0.5f) / (float)height;

	tvx2[pos] = tex2D(texTvx, x, y);
	tvy2[pos] = tex2D(texTvy, x, y);
}

void Stereo::FindWarpingVector(const float *u, const float *v, const float *tvx, const float *tvy,
	int w, int h, int s,
	float *tvx2, float *tvy2)
{
	dim3 threads(BlockWidth, BlockHeight);
	dim3 blocks(iDivUp(w, threads.x), iDivUp(h, threads.y));

	// mirror if a coordinate value is out-of-range
	texTvx.addressMode[0] = cudaAddressModeMirror;
	texTvx.addressMode[1] = cudaAddressModeMirror;
	texTvx.filterMode = cudaFilterModeLinear;
	texTvx.normalized = true;

	// mirror if a coordinate value is out-of-range
	texTvy.addressMode[0] = cudaAddressModeMirror;
	texTvy.addressMode[1] = cudaAddressModeMirror;
	texTvy.filterMode = cudaFilterModeLinear;
	texTvy.normalized = true;

	cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();

	cudaBindTexture2D(0, texTvx, tvx, w, h, s * sizeof(float));
	cudaBindTexture2D(0, texTvy, tvy, w, h, s * sizeof(float));

	FindWarpingVectorKernel << <blocks, threads >> > (u, v, w, h, s, tvx2, tvy2);
}

// **************************************************
// ** Compute Optical flow (u,v) for Fisheye Stereo
// **************************************************

__global__ void ComputeOpticalFlowVectorKernel(const float *dw, const float *tvx2, const float *tvy2,
	int width, int height, int stride,
	float *du, float *dv)
{
	const int ix = threadIdx.x + blockIdx.x * blockDim.x;
	const int iy = threadIdx.y + blockIdx.y * blockDim.y;

	const int pos = ix + iy * stride;

	if (ix >= width || iy >= height) return;

	du[pos] = dw[pos] * tvx2[pos];
	dv[pos] = dw[pos] * tvy2[pos];
}

void Stereo::ComputeOpticalFlowVector(const float *dw, const float *tvx2, const float *tvy2,
	int w, int h, int s,
	float *du, float *dv)
{
	dim3 threads(BlockWidth, BlockHeight);
	dim3 blocks(iDivUp(w, threads.x), iDivUp(h, threads.y));

	ComputeOpticalFlowVectorKernel << <blocks, threads >> > (dw, tvx2, tvy2, w, h, s, du, dv);
}