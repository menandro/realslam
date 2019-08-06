#include "stereo.h"

/// source image
texture<float, 2, cudaReadModeElementType> texSource;
/// tracked image
texture<float, 2, cudaReadModeElementType> texTarget;

__global__ void ComputeDerivativesKernel(int width, int height, int stride,
	float *Ix, float *Iy, float *Iz)
{
	const int ix = threadIdx.x + blockIdx.x * blockDim.x;
	const int iy = threadIdx.y + blockIdx.y * blockDim.y;

	const int pos = ix + iy * stride;

	if (ix >= width || iy >= height) return;

	float dx = 1.0f / (float)width;
	float dy = 1.0f / (float)height;

	float x = ((float)ix + 0.5f) * dx;
	float y = ((float)iy + 0.5f) * dy;

	float t0, t1;
	// x derivative
	t0 = tex2D(texSource, x - 2.0f * dx, y);
	t0 -= tex2D(texSource, x - 1.0f * dx, y) * 8.0f;
	t0 += tex2D(texSource, x + 1.0f * dx, y) * 8.0f;
	t0 -= tex2D(texSource, x + 2.0f * dx, y);
	t0 /= 12.0f;

	t1 = tex2D(texTarget, x - 2.0f * dx, y);
	t1 -= tex2D(texTarget, x - 1.0f * dx, y) * 8.0f;
	t1 += tex2D(texTarget, x + 1.0f * dx, y) * 8.0f;
	t1 -= tex2D(texTarget, x + 2.0f * dx, y);
	t1 /= 12.0f;

	Ix[pos] = (t0 + t1) * 0.5f;

	// t derivative
	Iz[pos] = tex2D(texTarget, x, y) - tex2D(texSource, x, y);

	// y derivative
	t0 = tex2D(texSource, x, y - 2.0f * dy);
	t0 -= tex2D(texSource, x, y - 1.0f * dy) * 8.0f;
	t0 += tex2D(texSource, x, y + 1.0f * dy) * 8.0f;
	t0 -= tex2D(texSource, x, y + 2.0f * dy);
	t0 /= 12.0f;

	t1 = tex2D(texTarget, x, y - 2.0f * dy);
	t1 -= tex2D(texTarget, x, y - 1.0f * dy) * 8.0f;
	t1 += tex2D(texTarget, x, y + 1.0f * dy) * 8.0f;
	t1 -= tex2D(texTarget, x, y + 2.0f * dy);
	t1 /= 12.0f;

	Iy[pos] = (t0 + t1) * 0.5f;
}

///CUDA CALL FUNCTIONS ***********************************************************
void Stereo::ComputeDerivatives(float *I0, float *I1,
	int w, int h, int s,
	float *Ix, float *Iy, float *Iz)
{
	dim3 threads(BlockWidth, BlockHeight);
	dim3 blocks(iDivUp(w, threads.x), iDivUp(h, threads.y));

	// mirror if a coordinate value is out-of-range
	texSource.addressMode[0] = cudaAddressModeMirror;
	texSource.addressMode[1] = cudaAddressModeMirror;
	texSource.filterMode = cudaFilterModeLinear;
	texSource.normalized = true;

	texTarget.addressMode[0] = cudaAddressModeMirror;
	texTarget.addressMode[1] = cudaAddressModeMirror;
	texTarget.filterMode = cudaFilterModeLinear;
	texTarget.normalized = true;

	//cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();

	cudaBindTexture2D(0, texSource, I0, w, h, s * sizeof(float));
	cudaBindTexture2D(0, texTarget, I1, w, h, s * sizeof(float));

	ComputeDerivativesKernel << < blocks, threads >> > (w, h, s, Ix, Iy, Iz);
}

