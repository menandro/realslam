#include "upsampling.h"

texture<float, 2, cudaReadModeElementType> texForGradient;

__global__ void PropagateColorOnlyKernel(float* grad, float* lidar, float* depthOut,
	int width, int height, int stride)
{
	int iy = blockIdx.y * blockDim.y + threadIdx.y;        // current row 
	int ix = blockIdx.x * blockDim.x + threadIdx.x;        // current column 

	if ((iy < height) && (ix < width))
	{
		int pos = ix + iy * stride;

		int maxRad = 5;
		int kernelSize = maxRad * 2 + 1;
		int shift = maxRad;

		// Find closest lidar point
		float dnearest = 0.0f;
		int dnearest_idx;
		float r0 = 10000.0f;
		for (int j = 0; j < kernelSize; j++) {
			for (int i = 0; i < kernelSize; i++) {
				int col = (ix + i - shift);
				int row = (iy + j - shift);

				if ((col >= 0) && (col < width) && (row >= 0) && (row < height)) {
					//col + stride*row
					float currLidar = lidar[col + stride * row];
					if (currLidar != 0.0f) {
						float r = sqrtf((ix - col)*(ix - col) + (iy - row)*(iy - row));
						if (r < r0) {
							r0 = r;
							dnearest_idx = col + stride * row;
							dnearest = currLidar;
						}
					}
				}
			}
		}

		// Propagation
		float sum = 0.0f;
		float count = 0.0f;
		int countPoint = 0;
		for (int j = 0; j < kernelSize; j++) {
			for (int i = 0; i < kernelSize; i++) {
				int col = (ix + i - shift);
				int row = (iy + j - shift);

				if ((col >= 0) && (col < width) && (row >= 0) && (row < height)) {
					//col + stride*row
					float currLidar = lidar[col + stride * row];
					if (currLidar != 0.0f) {
						countPoint++;
						float gs = 1.0f / (1.0f + sqrtf((ix - col)*(ix - col) + (iy - row)*(iy - row)));
						float gr = 1.0f / (1.0f + fabsf(dnearest - currLidar));
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
						count += gs * gr * (1.0f / (gmax + 0.001f));
					}
				}
			}
		}
		float propagatedDepth;
		if (count != 0.0f) {
			propagatedDepth = sum / count;
		}
		else {
			propagatedDepth = 0.0f;
		}

		depthOut[pos] = propagatedDepth;
	}
}


void lup::Upsampling::PropagateColorOnly(float* grad, float* lidar, float* depthOut)
{
	dim3 threads(BlockWidth, BlockHeight);
	dim3 blocks(iDivUp(width, threads.x), iDivUp(height, threads.y));
	PropagateColorOnlyKernel << < blocks, threads >> > (grad, lidar, depthOut, width, height, stride);
}


__global__ void ConvertDepthTo3DKernel(float* depth, float* X, float* Y, float* Z, float focal,
	int width, int height, int stride)
{
	int iy = blockIdx.y * blockDim.y + threadIdx.y;        // current row 
	int ix = blockIdx.x * blockDim.x + threadIdx.x;        // current column 

	if ((iy < height) && (ix < width))
	{
		int pos = ix + iy * stride;
		Z[pos] = depth[pos];
		X[pos] = Z[pos] * ((float)ix - (float)width / 2.0f) / focal;
		Y[pos] = Z[pos] * ((float)iy - (float)height / 2.0f) / focal;
	}
}

void lup::Upsampling::ConvertDepthTo3D(float* depth, float* X, float* Y, float* Z, float focal)
{
	dim3 threads(BlockWidth, BlockHeight);
	dim3 blocks(iDivUp(width, threads.x), iDivUp(height, threads.y));
	ConvertDepthTo3DKernel << < blocks, threads >> > (depth, X, Y, Z, focal,
		width, height, stride);
}

__global__ void ConvertDepthTo3DKernel(float* depth, float* X, float* Y, float* Z, float focal, float cx, float cy,
	int width, int height, int stride)
{
	int iy = blockIdx.y * blockDim.y + threadIdx.y;        // current row 
	int ix = blockIdx.x * blockDim.x + threadIdx.x;        // current column 

	if ((iy < height) && (ix < width))
	{
		int pos = ix + iy * stride;
		Z[pos] = depth[pos];
		X[pos] = Z[pos] * ((float)ix - cx) / focal;
		Y[pos] = Z[pos] * ((float)iy - cy) / focal;
	}
}

void lup::Upsampling::ConvertDepthTo3D(float* depth, float* X, float* Y, float* Z, float focal, float cx, float cy)
{
	dim3 threads(BlockWidth, BlockHeight);
	dim3 blocks(iDivUp(width, threads.x), iDivUp(height, threads.y));
	ConvertDepthTo3DKernel << < blocks, threads >> > (depth, X, Y, Z, focal, cx, cy,
		width, height, stride);
}

__global__ void ConvertDepthTo3DKernel(float* depth, float* X, float* Y, float* Z, uchar3* sem, float focal,
	int width, int height, int stride)
{
	int iy = blockIdx.y * blockDim.y + threadIdx.y;        // current row 
	int ix = blockIdx.x * blockDim.x + threadIdx.x;        // current column 

	if ((iy < height) && (ix < width))
	{
		int pos = ix + iy * stride;
		if ((sem[pos].x == 69) && (sem[pos].y == 129) && (sem[pos].z == 180)) {
			Z[pos] = 0.0f;
			X[pos] = 0.0f;
			Y[pos] = 0.0f;
		}
		else {
			Z[pos] = depth[pos];
			X[pos] = Z[pos] * (float)ix / focal;
			Y[pos] = Z[pos] * (float)iy / focal;
		}

	}
}

void lup::Upsampling::ConvertDepthTo3D(float* depth, float* X, float* Y, float* Z, uchar3* sem, float focal)
{
	dim3 threads(BlockWidth, BlockHeight);
	dim3 blocks(iDivUp(width, threads.x), iDivUp(height, threads.y));
	ConvertDepthTo3DKernel << < blocks, threads >> > (depth, X, Y, Z, sem, focal,
		width, height, stride);
}

__global__ void ConvertDepthTo3DKernel(float* depth, float* X, float* Y, float* Z, uchar3* sem, float focal, float cx, float cy,
	int width, int height, int stride)
{
	int iy = blockIdx.y * blockDim.y + threadIdx.y;        // current row 
	int ix = blockIdx.x * blockDim.x + threadIdx.x;        // current column 

	if ((iy < height) && (ix < width))
	{
		int pos = ix + iy * stride;
		if ((sem[pos].x == 69) && (sem[pos].y == 129) && (sem[pos].z == 180)) {
			Z[pos] = 0.0f;
			X[pos] = 0.0f;
			Y[pos] = 0.0f;
		}
		else {
			Z[pos] = depth[pos];
			X[pos] = Z[pos] * ((float)ix - cx) / focal;
			Y[pos] = Z[pos] * ((float)iy - cy) / focal;
		}

	}
}

void lup::Upsampling::ConvertDepthTo3D(float* depth, float* X, float* Y, float* Z, uchar3* sem, float focal, float cx, float cy)
{
	dim3 threads(BlockWidth, BlockHeight);
	dim3 blocks(iDivUp(width, threads.x), iDivUp(height, threads.y));
	ConvertDepthTo3DKernel << < blocks, threads >> > (depth, X, Y, Z, sem, focal, cx, cy,
		width, height, stride);
}

__global__ void ConvertDepthTo3DKernel(float* depth, float* X, float* Y, float* Z, float* grad, float focal,
	int width, int height, int stride)
{
	int iy = blockIdx.y * blockDim.y + threadIdx.y;        // current row 
	int ix = blockIdx.x * blockDim.x + threadIdx.x;        // current column 

	if ((iy < height) && (ix < width))
	{
		int pos = ix + iy * stride;
		if (grad[pos] < 0.05f) {
			Z[pos] = depth[pos];
			X[pos] = Z[pos] * (float)ix / focal;
			Y[pos] = Z[pos] * (float)iy / focal;
		}
		else {
			Z[pos] = 0.0f;
			X[pos] = 0.0f;
			Y[pos] = 0.0f;
		}
	}
}

void lup::Upsampling::ConvertDepthTo3D(float* depth, float* X, float* Y, float* Z, float* grad, float focal)
{
	dim3 threads(BlockWidth, BlockHeight);
	dim3 blocks(iDivUp(width, threads.x), iDivUp(height, threads.y));
	ConvertDepthTo3DKernel << < blocks, threads >> > (depth, X, Y, Z, grad, focal,
		width, height, stride);
}


__global__ void PropagateNearestNeighborKernel(float* grad, float* lidar, uchar3* sem, float* motionStereo, float* depthOut,
	int width, int height, int stride)
{
	int iy = blockIdx.y * blockDim.y + threadIdx.y;        // current row 
	int ix = blockIdx.x * blockDim.x + threadIdx.x;        // current column 

	if ((iy < height) && (ix < width))
	{
		int pos = ix + iy * stride;

		int maxRad = 10;
		int kernelSize = maxRad * 2 + 1;
		int shift = maxRad;

		// Find closest lidar point
		float dnearest = 0.0f;
		int dnearest_idx;
		float r0 = 10000.0f;
		for (int j = 0; j < kernelSize; j++) {
			for (int i = 0; i < kernelSize; i++) {
				int col = (ix + i - shift);
				int row = (iy + j - shift);

				if ((col >= 0) && (col < width) && (row >= 0) && (row < height)) {
					//col + stride*row
					float currLidar = lidar[col + stride * row];
					if (currLidar != 0.0f) {
						float r = sqrtf((ix - col)*(ix - col) + (iy - row)*(iy - row));
						if (r < r0) {
							r0 = r;
							dnearest_idx = col + stride * row;
							dnearest = currLidar;
						}
					}
				}
			}
		}

		depthOut[pos] = dnearest;
	}
}


void lup::Upsampling::PropagateNearestNeighbor(float* grad, float* lidar, uchar3* semantic, float* motionStereo, float* depthOut)
{
	dim3 threads(BlockWidth, BlockHeight);
	dim3 blocks(iDivUp(width, threads.x), iDivUp(height, threads.y));
	PropagateNearestNeighborKernel << < blocks, threads >> > (grad, lidar, semantic, motionStereo, depthOut, width, height, stride);
}

__global__ void GradientKernel(float* output, int width, int height, int stride) {
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


void lup::Upsampling::Gradient(float* input, float* output) {
	dim3 threads(BlockWidth, BlockHeight);
	dim3 blocks(iDivUp(width, threads.x), iDivUp(height, threads.y));

	// mirror if a coordinate value is out-of-range
	texForGradient.addressMode[0] = cudaAddressModeMirror;
	texForGradient.addressMode[1] = cudaAddressModeMirror;
	texForGradient.filterMode = cudaFilterModeLinear;
	texForGradient.normalized = true;
	cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
	cudaBindTexture2D(0, texForGradient, input, width, height, stride * sizeof(float));

	GradientKernel << < blocks, threads >> > (output, width, height, stride);
}