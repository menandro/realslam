#include "stereolite.h"

// Subtract I0 and I1 for error
__global__
void LitePlaneSweepGetErrorKernel(float* i0, float* i1, int width, int height, int stride, float *error)
{
	const int ix = threadIdx.x + blockIdx.x * blockDim.x;
	const int iy = threadIdx.y + blockIdx.y * blockDim.y;

	const int pos = ix + iy * stride;

	if (ix >= width || iy >= height) return;

	error[pos] = fabsf(i0[pos] - i1[pos]);
}


// Remove disparity of pixels whose error is very close to the mean error
__global__
void LitePlaneSweepMeanCleanup(float* error, float* meanError, float standardDev, float* disparity, float2* finalWarp,
	int width, int height, int stride)
{
	const int ix = threadIdx.x + blockIdx.x * blockDim.x;
	const int iy = threadIdx.y + blockIdx.y * blockDim.y;

	const int pos = ix + iy * stride;

	if (ix >= width || iy >= height) return;

	if ((meanError[pos] - error[pos]) > standardDev) {
		error[pos] = 0.0f;
		disparity[pos] = 0.0f;
		finalWarp[pos] = make_float2(0.0f, 0.0f);
	}
}

__global__
void LitePlaneSweepMeanCleanup(float* error, float* meanError, float standardDev, float* disparity,
	int width, int height, int stride)
{
	const int ix = threadIdx.x + blockIdx.x * blockDim.x;
	const int iy = threadIdx.y + blockIdx.y * blockDim.y;

	const int pos = ix + iy * stride;

	if (ix >= width || iy >= height) return;

	if ((meanError[pos] - error[pos]) > standardDev) {
		error[pos] = 0.0f;
		disparity[pos] = 0.0f;
	}
}


// Window-based SAD
__global__
void LitePlaneSweepCorrelationKernel(float* imError, float* disparity, int sweepDistance, int maxDisparity,
	int windowSize, int width, int height, int stride, float *error, float *meanError)
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
				currError += imError[col + stride * row];
				windowCount++;
			}
		}
	}
	currError = currError / windowCount;
	meanError[pos] = ((float)sweepDistance * meanError[pos] + currError) / ((float)sweepDistance + 1.0f);
	if (currError < error[pos]) {
		/*if (sweepDistance == maxDisparity) {
			error[pos] = 0.0f;
			disparity[pos] = 0.0f;
		}
		else {*/
			error[pos] = currError;
			disparity[pos] = (float)sweepDistance;
		//}
	}

}

void StereoLite::PlaneSweepCorrelation(float *i0, float *i1, float* disparity, int sweepDistance, int windowSize,
	int w, int h, int s, float *error)
{
	dim3 threads(BlockWidth, BlockHeight);
	dim3 blocks(iDivUp(w, threads.x), iDivUp(h, threads.y));

	//LitePlaneSweepCorrelationKernel << <blocks, threads >> > (i0, i1, disparity, sweepDistance, windowSize, w, h, s, error);
	LitePlaneSweepGetErrorKernel << < blocks, threads >> > (i0, i1, w, h, s, ps_errorHolder);
	LitePlaneSweepCorrelationKernel << <blocks, threads >> > (ps_errorHolder, disparity, sweepDistance, 
		planeSweepMaxDisparity, windowSize,	w, h, s, error, ps_meanError);
	LitePlaneSweepMeanCleanup << < blocks, threads >> > (error, ps_meanError, planeSweepStandardDev, disparity,
		w, h, s);
}


// Window-based SAD with warping vector fetch for left-right consistency calculation
__global__
void LitePlaneSweepCorrelationGetWarpKernel(float* imError, float* disparity, int sweepDistance, int maxDisparity,
	int windowSize, float2* currentWarp, float2 * finalWarp, float2* tv,
	int width, int height, int stride, float *error, float *meanError)
{
	const int ix = threadIdx.x + blockIdx.x * blockDim.x;
	const int iy = threadIdx.y + blockIdx.y * blockDim.y;

	const int pos = ix + iy * stride;

	if (ix >= width || iy >= height) return;

	currentWarp[pos].x = currentWarp[pos].x + tv[pos].x;
	currentWarp[pos].y = currentWarp[pos].y + tv[pos].y;

	float currError = 0.0f;
	int windowCount = 0;
	for (int j = 0; j < windowSize; j++) {
		for (int i = 0; i < windowSize; i++) {
			//get values
			int col = (ix + i - (windowSize - 1) / 2);
			int row = (iy + j - (windowSize - 1) / 2);
			if ((col >= 0) && (col < width) && (row >= 0) && (row < height)) {
				currError += imError[col + stride * row];
				windowCount++;
			}
		}
	}
	currError = currError / windowCount;
	meanError[pos] = ((float)sweepDistance * meanError[pos] + currError) / ((float)sweepDistance + 1.0f);
	if (currError < error[pos]) {
		/*if (sweepDistance == maxDisparity) {
			error[pos] = 0.0f;
			disparity[pos] = 0.0f;
			finalWarp[pos] = make_float2(0.0f, 0.0f);
		}
		else {*/
			error[pos] = currError;
			disparity[pos] = (float)sweepDistance;
			finalWarp[pos] = currentWarp[pos];
		//}
	}
}


void StereoLite::PlaneSweepCorrelationGetWarp(float *i0, float *i1, float* disparity, int sweepDistance, int windowSize,
	float2* currentWarp, float2* finalWarp, float2 * translationVector, int w, int h, int s, float *error)
{
	dim3 threads(BlockWidth, BlockHeight);
	dim3 blocks(iDivUp(w, threads.x), iDivUp(h, threads.y));

	/*LitePlaneSweepCorrelationGetWarpKernel << <blocks, threads >> > (i0, i1, disparity, sweepDistance, windowSize, 
		currentWarp, finalWarp, translationVector, w, h, s, error);*/
	LitePlaneSweepGetErrorKernel << < blocks, threads >> > (i0, i1, w, h, s, ps_errorHolder);
	LitePlaneSweepCorrelationGetWarpKernel << <blocks, threads >> > (ps_errorHolder, disparity, sweepDistance, 
		planeSweepMaxDisparity, windowSize, currentWarp, finalWarp, translationVector, w, h, s, error, ps_meanError);
	LitePlaneSweepMeanCleanup << < blocks, threads >> > (error, ps_meanError, planeSweepStandardDev, disparity, finalWarp, 
		w, h, s);
}


// Left to Right Consistency
texture<float, cudaTextureType2D, cudaReadModeElementType> disparityTex;

__global__
void LiteLeftRightConsistencyKernel(float *disparityForward, float2* warpingVector, float* leftRightDiff,
	float epsilon, float* disparityFinal, float2* finalWarpForward, int width, int height, int stride)
{
	const int ix = threadIdx.x + blockIdx.x * blockDim.x;
	const int iy = threadIdx.y + blockIdx.y * blockDim.y;

	const int pos = ix + iy * stride;

	if (ix >= width || iy >= height) return;

	float2 warpUV = warpingVector[pos];
	int windowSize = 3;
	bool isConsistent = true;
	float currDiff = 0.0f;
	int windowCnt = 0;
	for (int j = 0; j < windowSize; j++) {
		for (int i = 0; i < windowSize; i++) {
			//get values
			int col = (ix + i - (windowSize - 1) / 2);
			int row = (iy + j - (windowSize - 1) / 2);
			if ((col >= 0) && (col < width) && (row >= 0) && (row < height)) {
				/*if (leftRightDiff[col + stride * row] > epsilon) {
					isConsistent = false;
				}*/
				currDiff += leftRightDiff[col + stride * row];
				windowCnt++;
			}
		}
	}
	currDiff = currDiff / (float)windowCnt;
	if (currDiff > epsilon) {
		isConsistent = false;
	}
	if (!isConsistent){
		disparityFinal[pos] = 0.0f;
		finalWarpForward[pos] = make_float2(0.0f, 0.0f);
	}
	else {
		disparityFinal[pos] = disparityForward[pos];
		finalWarpForward[pos] = warpUV;
	}
}

__global__
void LiteLeftRightDiffKernel(float *disparityForward, float2* warpingVector, float* leftRightDiff, 
	int width, int height, int stride)
{
	const int ix = threadIdx.x + blockIdx.x * blockDim.x;
	const int iy = threadIdx.y + blockIdx.y * blockDim.y;

	const int pos = ix + iy * stride;

	if (ix >= width || iy >= height) return;

	float2 warpUV = warpingVector[pos];
	float x = ((float)ix + warpUV.x + 0.5f) / (float)width;
	float y = ((float)iy + warpUV.y + 0.5f) / (float)height;
	float dx = 1.0f / (float)width;
	float dy = 1.0f / (float)height;

	float dispBackward = tex2D(disparityTex, x, y);
	leftRightDiff[pos] = abs(dispBackward - disparityForward[pos]);
}

void StereoLite::LeftRightConsistency(float *disparityForward, float* disparityBackward, float2* warpingVector, 
	float epsilon, float* disparityFinal, float2* finalWarpForward,
	int w, int h, int s)
{
	dim3 threads(BlockWidth, BlockHeight);
	dim3 blocks(iDivUp(w, threads.x), iDivUp(h, threads.y));

	// mirror if a coordinate value is out-of-range
	disparityTex.addressMode[0] = cudaAddressModeMirror;
	disparityTex.addressMode[1] = cudaAddressModeMirror;
	disparityTex.filterMode = cudaFilterModeLinear;
	disparityTex.normalized = true;

	cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();

	cudaBindTexture2D(0, disparityTex, disparityBackward, w, h, s * sizeof(float));

	LiteLeftRightDiffKernel << < blocks, threads >> > (disparityForward, warpingVector, ps_leftRightDiff, w, h, s);
	LiteLeftRightConsistencyKernel << <blocks, threads >> > (disparityForward, warpingVector, ps_leftRightDiff,
		epsilon, disparityFinal, finalWarpForward, w, h, s);
}



// [Hirata] Upsampling/Propagation
__global__ void LitePropagateColorOnlyKernel(float* grad, float* lidar, float2* warpUV, float2 * warpUVOut, float* depthOut, int radius,
	int width, int height, int stride)
{
	int iy = blockIdx.y * blockDim.y + threadIdx.y;        // current row 
	int ix = blockIdx.x * blockDim.x + threadIdx.x;        // current column 

	if ((iy < height) && (ix < width))
	{
		int pos = ix + iy * stride;

		int maxRad = radius;
		int kernelSize = maxRad * 2 + 1;
		int shift = maxRad;

		// Find closest lidar point
		float dnearest = 0.0f;
		float2 uvnearest;
		int dnearest_idx;
		float r0 = 10000.0f;
		for (int j = 0; j < kernelSize; j++) {
			for (int i = 0; i < kernelSize; i++) {
				int col = (ix + i - shift);
				int row = (iy + j - shift);

				if ((col >= 0) && (col < width) && (row >= 0) && (row < height)) {
					//col + stride*row
					float currLidar = lidar[col + stride * row];
					float2 currUV = warpUV[col + stride * row];
					if (currLidar != 0.0f) {
						float r = sqrtf((ix - col)*(ix - col) + (iy - row)*(iy - row));
						if (r < r0) {
							r0 = r;
							dnearest_idx = col + stride * row;
							dnearest = currLidar;
							uvnearest = currUV;
						}
					}
				}
			}
		}

		// Propagation
		float sum = 0.0f;
		float sumU = 0.0f;
		float sumV = 0.0f;

		float count = 0.0f;
		int countPoint = 0;
		for (int j = 0; j < kernelSize; j++) {
			for (int i = 0; i < kernelSize; i++) {
				int col = (ix + i - shift);
				int row = (iy + j - shift);

				if ((col >= 0) && (col < width) && (row >= 0) && (row < height)) {
					//col + stride*row
					float currLidar = lidar[col + stride * row];
					float2 currUV = warpUV[col + stride * row];
					if (currLidar != 0.0f) {
						countPoint++;
						float gs = 1.0f / (1.0f + sqrtf((ix - col)*(ix - col) + (iy - row)*(iy - row)));
						float gr = 1.0f / (1.0f + fabsf(dnearest - currLidar));
						float grU = 1.0f / (1.0f + fabsf(uvnearest.x - currUV.x));
						float grV = 1.0f / (1.0f + fabsf(uvnearest.y - currUV.y));
						// Find maximum gradient in between the current lidar point and the current pixel
						float gmax = grad[pos];
						// y-direction
						if (iy < row) {
							for (int gy = iy; gy <= row; gy++) {
								if (grad[ix + stride * gy] > gmax) {
									gmax = grad[ix + stride * gy];
								}
							}
						}
						else if (iy > row) {
							for (int gy = row; gy <= iy; gy++) {
								if (grad[ix + stride * gy] > gmax) {
									gmax = grad[ix + stride * gy];
								}
							}
						}
						// x-direction
						if (ix < col) {
							for (int gx = ix; gx <= col; gx++) {
								if (grad[gx + stride * iy] > gmax) {
									gmax = grad[gx + stride * iy];
								}
							}
						}
						else if (ix > col) {
							for (int gx = col; gx <= ix; gx++) {
								if (grad[gx + stride * iy] > gmax) {
									gmax = grad[gx + stride * iy];
								}
							}
						}
						sum += currLidar * gs * gr * (1.0f / (gmax + 0.001f));
						sumU += warpUV[col + stride * row].x * gs * grU * (1.0f / (gmax + 0.001f));
						sumV += warpUV[col + stride * row].y * gs * grV * (1.0f / (gmax + 0.001f));
						count += gs * gr * (1.0f / (gmax + 0.001f));
					}
				}
			}
		}
		float propagatedDepth;
		float2 propagatedUV;
		if (count != 0.0f) {
			propagatedDepth = sum / count;
			propagatedUV.x = sumU / count;
			propagatedUV.y = sumV / count;
		}
		else {
			propagatedDepth = 0.0f;
			propagatedUV.x = 0.0f;
			propagatedUV.y = 0.0f;
		}

		depthOut[pos] = propagatedDepth;
	}
}


void StereoLite::PropagateColorOnly(float* grad, float* lidar, float2 * warpUV, float2* warpUVOut, float* depthOut, int radius)
{
	dim3 threads(BlockWidth, BlockHeight);
	dim3 blocks(iDivUp(width, threads.x), iDivUp(height, threads.y));
	LitePropagateColorOnlyKernel << < blocks, threads >> > (grad, lidar, warpUV, warpUVOut, depthOut, radius, width, height, stride);
}

texture<float, 2, cudaReadModeElementType> texForGradient;

__global__ void LiteGradientKernel(float* output, int width, int height, int stride) {
	int iy = blockIdx.y * blockDim.y + threadIdx.y;        // current row 
	int ix = blockIdx.x * blockDim.x + threadIdx.x;        // current column 

	if ((iy < height) && (ix < width))
	{
		int pos = ix + iy * stride;

		float dx = 1.0f / (float)width;
		float dy = 1.0f / (float)height;

		float x = ((float)ix + 0.5f) * dx;
		float y = ((float)iy + 0.5f) * dy;

		float2 grad;
		float t0;
		// x derivative
		t0 = tex2D(texForGradient, x + 1.0f * dx, y);
		t0 -= tex2D(texForGradient, x, y);
		t0 = tex2D(texForGradient, x + 1.0f * dx, y + 1.0f * dy);
		t0 -= tex2D(texForGradient, x, y + 1.0f * dy);
		grad.x = t0;

		// y derivative
		t0 = tex2D(texForGradient, x, y + 1.0f * dy);
		t0 -= tex2D(texForGradient, x, y);
		t0 = tex2D(texForGradient, x + 1.0f * dx, y + 1.0f * dy);
		t0 -= tex2D(texForGradient, x + 1.0f * dx, y);
		grad.y = t0;

		output[pos] = sqrtf(grad.x * grad.x + grad.y * grad.y);
	}
}


void StereoLite::Gradient(float* input, float* output) {
	dim3 threads(BlockWidth, BlockHeight);
	dim3 blocks(iDivUp(width, threads.x), iDivUp(height, threads.y));

	// mirror if a coordinate value is out-of-range
	texForGradient.addressMode[0] = cudaAddressModeMirror;
	texForGradient.addressMode[1] = cudaAddressModeMirror;
	texForGradient.filterMode = cudaFilterModeLinear;
	texForGradient.normalized = true;
	cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
	cudaBindTexture2D(0, texForGradient, input, width, height, stride * sizeof(float));

	LiteGradientKernel << < blocks, threads >> > (output, width, height, stride);
}


__global__ void GetMaskPositiveKernel(float* input, float* output, int width, int height, int stride) {
	const int ix = threadIdx.x + blockIdx.x * blockDim.x;
	const int iy = threadIdx.y + blockIdx.y * blockDim.y;

	const int pos = ix + iy * stride;

	if (ix >= width || iy >= height) return;
	
	if (input[pos] > 0.0f) {
		output[pos] = 1.0f;
	}
	else {
		output[pos] = 0.0f;
	}
}

__global__ void GetMaskNegativeKernel(float* input, float* output, int width, int height, int stride) {
	const int ix = threadIdx.x + blockIdx.x * blockDim.x;
	const int iy = threadIdx.y + blockIdx.y * blockDim.y;

	const int pos = ix + iy * stride;

	if (ix >= width || iy >= height) return;

	if (input[pos] > 0.0f) {
		output[pos] = 0.0f;
	}
	else {
		output[pos] = 1.0f;
	}
}


void StereoLite::GetMask(float* input, float* output, bool isPositive, int w, int h, int s) {
	dim3 threads(BlockWidth, BlockHeight);
	dim3 blocks(iDivUp(width, threads.x), iDivUp(height, threads.y));

	if (isPositive) {
		GetMaskPositiveKernel << < blocks, threads >> > (input, output, w, h, s);
	}
	else {
		GetMaskNegativeKernel << < blocks, threads >> > (input, output, w, h, s);
	}
}



















// OLD KERNELS
__global__
void LitePlaneSweepCorrelationKernel(float* i0, float* i1, float* disparity, int sweepDistance,
	int windowSize, int width, int height, int stride, float *error)
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

__global__
void LitePlaneSweepCorrelationGetWarpKernel(float* i0, float* i1, float* disparity, int sweepDistance,
	int windowSize, float2* currentWarp, float2 * finalWarp, float2* tv,
	int width, int height, int stride, float *error)
{
	const int ix = threadIdx.x + blockIdx.x * blockDim.x;
	const int iy = threadIdx.y + blockIdx.y * blockDim.y;

	const int pos = ix + iy * stride;

	if (ix >= width || iy >= height) return;

	currentWarp[pos].x = currentWarp[pos].x + tv[pos].x;
	currentWarp[pos].y = currentWarp[pos].y + tv[pos].y;

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
		finalWarp[pos] = currentWarp[pos];
	}
}