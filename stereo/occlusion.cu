#include "stereo.h"

/// image to warp
texture<float, 2, cudaReadModeElementType> texReference;

__global__ void OcclusionCheckKernel(float * baseDepth, float threshold, float *u, float *v, 
	int width, int height, int stride, float *out)
{
	const int ix = threadIdx.x + blockIdx.x * blockDim.x;
	const int iy = threadIdx.y + blockIdx.y * blockDim.y;

	if (ix >= width || iy >= height) return;

	const int pos = ix + iy * stride;

	float x = ((float)ix + u[pos] + 0.5f) / (float)width;
	float y = ((float)iy + v[pos] + 0.5f) / (float)height;

	float referenceDepth = tex2D(texReference, x, y);
	if (fabsf(referenceDepth - baseDepth[pos]) > threshold) {
		out[pos] = 0.0f;
	}
	else {
		out[pos] = baseDepth[pos];
	}
}

void Stereo::OcclusionCheck(float* wForward, float* wBackward, float threshold, float *u, float *v,
	int w, int h, int s, float* wFinal)
{
	dim3 threads(BlockWidth, BlockHeight);
	dim3 blocks(iDivUp(w, threads.x), iDivUp(h, threads.y));

	// mirror if a coordinate value is out-of-range
	texReference.addressMode[0] = cudaAddressModeMirror;
	texReference.addressMode[1] = cudaAddressModeMirror;
	texReference.filterMode = cudaFilterModeLinear;
	texReference.normalized = true;

	cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();

	cudaBindTexture2D(0, texReference, wBackward, w, h, s * sizeof(float));

	OcclusionCheckKernel << <blocks, threads >> > (wForward, threshold, u, v, w, h, s, wFinal);
}