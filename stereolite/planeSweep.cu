#include "stereolite.h"

__global__
void LitePlaneSweepCorrelationKernel(float* i0, float* i1, float* disparity, int sweepDistance,
	int windowSize,
	int width, int height, int stride,
	float *error)
{
	const int ix = threadIdx.x + blockIdx.x * blockDim.x;
	const int iy = threadIdx.y + blockIdx.y * blockDim.y;

	const int pos = ix + iy * stride;

	if (ix >= width || iy >= height) return;

	float currError = 0.0f;
	int windowCount = 0;
	for (int j = 0; j < windowSize; j++) {
		for (int i = 0; i < windowSize; i++) {
			//get values
			int col = (ix + i - (windowSize - 1) / 2);
			int row = (iy + j - (windowSize - 1) / 2);
			if ((col >= 0) && (col < width) && (row >= 0) && (row < height)) {
				currError += fabsf(i0[col + stride * row] - i1[col + stride * row]);
				windowCount++;
			}
		}
	}
	currError = currError / windowCount;
	if (currError < error[pos]) {
		error[pos] = currError;
		disparity[pos] = (float)sweepDistance;
	}

}

void StereoLite::PlaneSweepCorrelation(float *i0, float *i1, float* disparity, int sweepDistance, int windowSize,
	int w, int h, int s,
	float *error)
{
	dim3 threads(BlockWidth, BlockHeight);
	dim3 blocks(iDivUp(w, threads.x), iDivUp(h, threads.y));

	LitePlaneSweepCorrelationKernel << <blocks, threads >> > (i0, i1, disparity, sweepDistance, windowSize, w, h, s, error);
}