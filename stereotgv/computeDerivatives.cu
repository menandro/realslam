#include "stereotgv.h"

texture<float, cudaTextureType2D, cudaReadModeElementType> texI0;
texture<float, cudaTextureType2D, cudaReadModeElementType> texI1;

__global__ void TgvComputeDerivativesKernel(int width, int height, int stride,
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
	t0 = tex2D(texI0, x - 2.0f * dx, y);
	t0 -= tex2D(texI0, x - 1.0f * dx, y) * 8.0f;
	t0 += tex2D(texI0, x + 1.0f * dx, y) * 8.0f;
	t0 -= tex2D(texI0, x + 2.0f * dx, y);
	t0 /= 12.0f;

	t1 = tex2D(texI1, x - 2.0f * dx, y);
	t1 -= tex2D(texI1, x - 1.0f * dx, y) * 8.0f;
	t1 += tex2D(texI1, x + 1.0f * dx, y) * 8.0f;
	t1 -= tex2D(texI1, x + 2.0f * dx, y);
	t1 /= 12.0f;

	Ix[pos] = (t0 + t1) * 0.5f;

	// t derivative
	Iz[pos] = tex2D(texI1, x, y) - tex2D(texI0, x, y);

	// y derivative
	t0 = tex2D(texI0, x, y - 2.0f * dy);
	t0 -= tex2D(texI0, x, y - 1.0f * dy) * 8.0f;
	t0 += tex2D(texI0, x, y + 1.0f * dy) * 8.0f;
	t0 -= tex2D(texI0, x, y + 2.0f * dy);
	t0 /= 12.0f;

	t1 = tex2D(texI1, x, y - 2.0f * dy);
	t1 -= tex2D(texI1, x, y - 1.0f * dy) * 8.0f;
	t1 += tex2D(texI1, x, y + 1.0f * dy) * 8.0f;
	t1 -= tex2D(texI1, x, y + 2.0f * dy);
	t1 /= 12.0f;

	Iy[pos] = (t0 + t1) * 0.5f;
}

///CUDA CALL FUNCTIONS ***********************************************************
void StereoTgv::ComputeDerivatives(float *I0, float *I1,
	int w, int h, int s,
	float *Ix, float *Iy, float *Iz)
{
	dim3 threads(BlockWidth, BlockHeight);
	dim3 blocks(iDivUp(w, threads.x), iDivUp(h, threads.y));

	// mirror if a coordinate value is out-of-range
	texI0.addressMode[0] = cudaAddressModeMirror;
	texI0.addressMode[1] = cudaAddressModeMirror;
	texI0.filterMode = cudaFilterModeLinear;
	texI0.normalized = true;

	texI1.addressMode[0] = cudaAddressModeMirror;
	texI1.addressMode[1] = cudaAddressModeMirror;
	texI1.filterMode = cudaFilterModeLinear;
	texI1.normalized = true;

	//cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();

	cudaBindTexture2D(0, texI0, I0, w, h, s * sizeof(float));
	cudaBindTexture2D(0, texI1, I1, w, h, s * sizeof(float));

	TgvComputeDerivativesKernel << < blocks, threads >> > (w, h, s, Ix, Iy, Iz);
}


//****************************************
// Fisheye Stereo Census Transform
//****************************************
__global__
void TgvComputeCensusFisheyeKernel(float * I0, float * I1, float eps, int width, int height, int stride,
	float* Iz)
{
	const int ix = threadIdx.x + blockIdx.x * blockDim.x;
	const int iy = threadIdx.y + blockIdx.y * blockDim.y;

	const int pos = ix + iy * stride;

	if (ix >= width || iy >= height) return;

	float centerPix0 = I0[pos];
	float centerPix1 = I1[pos];
	// 3x3 window
	float hamming = 0.0f;
	int windowRadius = 3;
	int pixCount = 0;
	for (int j = -windowRadius; j <= windowRadius; j++) {
		for (int i = -windowRadius; i <= windowRadius; i++) {
			//get values
			int col = (ix + i);
			int row = (iy + j);
			if ((col >= 0) && (col < width) && (row >= 0) && (row < height)) {
				float currPix0 = I0[col + row * stride];
				float currPix1 = I1[col + row * stride];
				float appendBit0, appendBit1;
				if ((j != 0) && (i != 0)) {
					if ((centerPix0 - currPix0) > eps) appendBit0 = 0.0f;
					else if (fabs(centerPix0 - currPix0) < eps) appendBit0 = 1.0f;
					else appendBit0 = 2.0f;

					if ((centerPix1 - currPix1) > eps) appendBit1 = 0.0f;
					else if (fabs(centerPix1 - currPix1) < eps) appendBit1 = 1.0f;
					else appendBit1 = 2.0f;	
					
					if (appendBit0 != appendBit1) {
						hamming = hamming + 1.0f;
					}
				}
				pixCount++;
			}
		}
	}

	Iz[pos] = hamming / (float)pixCount;
	//Iz[pos] = (2.0f * hamming / (float)pixCount) - 1.0f;
}


void StereoTgv::ComputeCensusFisheye(float* I0, float* I1, float eps,
	int w, int h, int s, float* Iz)
{
	dim3 threads(BlockWidth, BlockHeight);
	dim3 blocks(iDivUp(w, threads.x), iDivUp(h, threads.y));

	TgvComputeCensusFisheyeKernel << < blocks, threads >> > (I0, I1, eps, w, h, s, Iz);
}


//****************************************
// Fisheye Stereo Census Transform Derivative
//****************************************
__global__
void TgvComputeCensusDerivativesFisheyeKernel(float2* vector, int width, int height, int stride,
	float* Iw)
{
	const int ix = threadIdx.x + blockIdx.x * blockDim.x;
	const int iy = threadIdx.y + blockIdx.y * blockDim.y;

	const int pos = ix + iy * stride;

	if (ix >= width || iy >= height) return;

	float vx = vector[pos].x;
	float vy = vector[pos].y;
	float r = sqrtf(vx * vx + vy * vy);

	// Normalize because pyramid sampling ruins normality
	float dx = (vx / r) / (float)width;
	float dy = (vy / r) / (float)height;

	float x = ((float)ix + 0.5f) / (float)width; 
	float y = ((float)iy + 0.5f) / (float)height;

	float t0;
	// curve w derivative
	t0 = tex2D(texI0, x - 2.0f * dx, y - 2.0f * dy);
	t0 -= tex2D(texI0, x - 1.0f * dx, y - 1.0f * dy) * 8.0f;
	t0 += tex2D(texI0, x + 1.0f * dx, y + 1.0f * dy) * 8.0f;
	t0 -= tex2D(texI0, x + 2.0f * dx, y + 2.0f * dy);
	t0 /= 12.0f;
	
	Iw[pos] = t0;
}

///CUDA CALL FUNCTIONS ***********************************************************
void StereoTgv::ComputeCensusDerivativesFisheye(float* Iz, float2* vector,
	int w, int h, int s, float* Iw)
{
	dim3 threads(BlockWidth, BlockHeight);
	dim3 blocks(iDivUp(w, threads.x), iDivUp(h, threads.y));

	// mirror if a coordinate value is out-of-range
	texI0.addressMode[0] = cudaAddressModeMirror;
	texI0.addressMode[1] = cudaAddressModeMirror;
	texI0.filterMode = cudaFilterModeLinear;
	texI0.normalized = true;
	
	cudaBindTexture2D(0, texI0, Iz, w, h, s * sizeof(float));

	TgvComputeCensusDerivativesFisheyeKernel << < blocks, threads >> > (vector, w, h, s, Iw);
}



//****************************************
// Fisheye Stereo 1D Derivative
//****************************************
__global__
void TgvComputeDerivativesFisheyeKernel(float2 * vector, int width, int height, int stride,
	float *Iw, float *Iz)
{
	const int ix = threadIdx.x + blockIdx.x * blockDim.x;
	const int iy = threadIdx.y + blockIdx.y * blockDim.y;

	const int pos = ix + iy * stride;

	if (ix >= width || iy >= height) return;

	float vx = vector[pos].x;
	float vy = vector[pos].y;
	float r = sqrtf(vx * vx + vy * vy);

	// Normalize because pyramid sampling ruins normality
	float dx = (vx / r) / (float)width;
	float dy = (vy / r) / (float)height;

	float x = ((float)ix + 0.5f) / (float)width;
	float y = ((float)iy + 0.5f) / (float)height;

	float t0;
	// curve w derivative
	t0 = tex2D(texI0, x - 2.0f * dx, y - 2.0f * dy);
	t0 -= tex2D(texI0, x - 1.0f * dx, y - 1.0f * dy) * 8.0f;
	t0 += tex2D(texI0, x + 1.0f * dx, y + 1.0f * dy) * 8.0f;
	t0 -= tex2D(texI0, x + 2.0f * dx, y + 2.0f * dy);
	t0 /= 12.0f;

	float t1;
	t1 = tex2D(texI1, x - 2.0f * dx, y - 2.0f * dy);
	t1 -= tex2D(texI1, x - 1.0f * dx, y - 1.0f * dy) * 8.0f;
	t1 += tex2D(texI1, x + 1.0f * dx, y + 1.0f * dy) * 8.0f;
	t1 -= tex2D(texI1, x + 2.0f * dx, y + 2.0f * dy);
	t1 /= 12.0f;

	Iw[pos] = (t0 + t1) * 0.5f;

	// t derivative
	Iz[pos] = tex2D(texI1, x, y) - tex2D(texI0, x, y);
}

///CUDA CALL FUNCTIONS ***********************************************************
void StereoTgv::ComputeDerivativesFisheye(float *I0, float *I1, float2 *vector,
	int w, int h, int s, float *Iw, float *Iz)
{
	dim3 threads(BlockWidth, BlockHeight);
	dim3 blocks(iDivUp(w, threads.x), iDivUp(h, threads.y));

	// mirror if a coordinate value is out-of-range
	texI0.addressMode[0] = cudaAddressModeMirror;
	texI0.addressMode[1] = cudaAddressModeMirror;
	texI0.filterMode = cudaFilterModeLinear;
	texI0.normalized = true;

	texI1.addressMode[0] = cudaAddressModeMirror;
	texI1.addressMode[1] = cudaAddressModeMirror;
	texI1.filterMode = cudaFilterModeLinear;
	texI1.normalized = true;

	//cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();

	cudaBindTexture2D(0, texI0, I0, w, h, s * sizeof(float));
	cudaBindTexture2D(0, texI1, I1, w, h, s * sizeof(float));

	TgvComputeDerivativesFisheyeKernel << < blocks, threads >> > (vector, w, h, s, Iw, Iz);
}


//****************************************
// Fisheye Stereo 1D Derivative Equidistant Model
//****************************************
__global__
void TgvComputeDerivativesFisheyeEquidistantKernel(float focal, float cx, float cy, float tx, float ty, float tz, 
	int width, int height, int stride,
	float *Iw, float *Iz)
{
	const int ix = threadIdx.x + blockIdx.x * blockDim.x;
	const int iy = threadIdx.y + blockIdx.y * blockDim.y;

	const int pos = ix + iy * stride;

	if (ix >= width || iy >= height) return;

	float u0 = ix + 1;
	float v0 = iy + 1;

	//float x = ((float)ix + warpUV[pos].x + 0.5f) / (float)width;
	//float y = ((float)iy + warpUV[pos].y + 0.5f) / (float)height;

	// Create arbitrary surface
	float xprime0 = (u0 - cx);
	float yprime0 = (v0 - cy);
	float theta = sqrtf(xprime0*xprime0 + yprime0 * yprime0) / focal;
	float Xradius = 1.0f;
	float Z = Xradius * cosf(theta);
	float phi = atan2f(yprime0, xprime0);
	float X = Xradius * sinf(theta) * cosf(phi);
	float Y = Xradius * sinf(theta) * sinf(phi);

	// Forward vector
	float XX = X + tx;
	float YY = Y + ty;
	float ZZ = Z + tz;
	float XXradius = sqrtf(XX*XX + YY * YY);
	float theta2 = atan2f(XXradius, ZZ);
	float alpha2 = focal * theta2;
	float phi2 = atan2(YY, XX);
	float xprime1 = alpha2 * cosf(phi2);
	float yprime1 = alpha2 * sinf(phi2);
	float u1 = xprime1 + cx;
	float v1 = yprime1 + cy;
	float vectorxforward = u1 - u0;
	float vectoryforward = v1 - v0;

	// Backward vector
	XX = X - tx;
	YY = Y - ty;
	ZZ = Z - tz;
	XXradius = sqrt(XX*XX + YY * YY);
	theta2 = atan2(XXradius, ZZ);
	alpha2 = focal * theta2;
	phi2 = atan2f(YY, XX);
	xprime1 = alpha2 * cosf(phi2);
	yprime1 = alpha2 * sinf(phi2);
	u1 = xprime1 + cx;
	v1 = yprime1 + cy;
	float vectorxbackward = u1 - u0;
	float vectorybackward = v1 - v0;

	// Solve trajectory vector
	float vectorx = 0.5f*(vectorxforward - vectorxbackward);
	float vectory = 0.5f*(vectoryforward - vectorybackward);
	// Normalize
	float magnitude = sqrt(vectorx*vectorx + vectory * vectory);
	float dx = (vectorx / magnitude) / (float)width;
	float dy = (vectory / magnitude) / (float)height;

	// Normalize because pyramid sampling ruins normality
	/*float dx = (vx / r) / (float)width;
	float dy = (vy / r) / (float)height;*/

	float x = ((float)ix + 0.5f) / (float)width;
	float y = ((float)iy + 0.5f) / (float)height;

	float t0;
	// curve w derivative
	t0 = tex2D(texI0, x - 2.0f * dx, y - 2.0f * dy);
	t0 -= tex2D(texI0, x - 1.0f * dx, y - 1.0f * dy) * 8.0f;
	t0 += tex2D(texI0, x + 1.0f * dx, y + 1.0f * dy) * 8.0f;
	t0 -= tex2D(texI0, x + 2.0f * dx, y + 2.0f * dy);
	t0 /= 12.0f;

	float t1;
	t1 = tex2D(texI1, x - 2.0f * dx, y - 2.0f * dy);
	t1 -= tex2D(texI1, x - 1.0f * dx, y - 1.0f * dy) * 8.0f;
	t1 += tex2D(texI1, x + 1.0f * dx, y + 1.0f * dy) * 8.0f;
	t1 -= tex2D(texI1, x + 2.0f * dx, y + 2.0f * dy);
	t1 /= 12.0f;

	Iw[pos] = (t0 + t1) * 0.5f;

	// t derivative
	Iz[pos] = tex2D(texI1, x, y) - tex2D(texI0, x, y);
}

///CUDA CALL FUNCTIONS ***********************************************************
void StereoTgv::ComputeDerivativesFisheyeEquidistant(float *I0, float *I1,
	float focal, float cx, float cy, float tx, float ty, float tz,
	int w, int h, int s, float *Iw, float *Iz)
{
	dim3 threads(BlockWidth, BlockHeight);
	dim3 blocks(iDivUp(w, threads.x), iDivUp(h, threads.y));

	// mirror if a coordinate value is out-of-range
	texI0.addressMode[0] = cudaAddressModeMirror;
	texI0.addressMode[1] = cudaAddressModeMirror;
	texI0.filterMode = cudaFilterModeLinear;
	texI0.normalized = true;

	texI1.addressMode[0] = cudaAddressModeMirror;
	texI1.addressMode[1] = cudaAddressModeMirror;
	texI1.filterMode = cudaFilterModeLinear;
	texI1.normalized = true;

	//cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();

	cudaBindTexture2D(0, texI0, I0, w, h, s * sizeof(float));
	cudaBindTexture2D(0, texI1, I1, w, h, s * sizeof(float));

	TgvComputeDerivativesFisheyeEquidistantKernel << < blocks, threads >> > (focal, cx, cy, tx, ty, tz,
		w, h, s, Iw, Iz);
}

//****************************************
// Fisheye Stereo 1D Derivative MASKED
//****************************************
__global__
void TgvComputeDerivativesFisheyeMaskedKernel(float2 * vector, float* mask, int width, int height, int stride,
	float *Iw, float *Iz)
{
	const int ix = threadIdx.x + blockIdx.x * blockDim.x;
	const int iy = threadIdx.y + blockIdx.y * blockDim.y;

	if ((iy >= height) && (ix >= width)) return;
	int pos = ix + iy * stride;
	if (mask[pos] == 0.0f) return;
	
	float vx = vector[pos].x;
	float vy = vector[pos].y;
	float r = sqrtf(vx * vx + vy * vy);

	// Normalize because pyramid sampling ruins normality
	float dx = (vx / r) / (float)width;
	float dy = (vy / r) / (float)height;

	float x = ((float)ix + 0.5f) / (float)width;
	float y = ((float)iy + 0.5f) / (float)height;

	float t0;
	// curve w derivative
	t0 = tex2D(texI0, x - 2.0f * dx, y - 2.0f * dy);
	t0 -= tex2D(texI0, x - 1.0f * dx, y - 1.0f * dy) * 8.0f;
	t0 += tex2D(texI0, x + 1.0f * dx, y + 1.0f * dy) * 8.0f;
	t0 -= tex2D(texI0, x + 2.0f * dx, y + 2.0f * dy);
	t0 /= 12.0f;

	float t1;
	t1 = tex2D(texI1, x - 2.0f * dx, y - 2.0f * dy);
	t1 -= tex2D(texI1, x - 1.0f * dx, y - 1.0f * dy) * 8.0f;
	t1 += tex2D(texI1, x + 1.0f * dx, y + 1.0f * dy) * 8.0f;
	t1 -= tex2D(texI1, x + 2.0f * dx, y + 2.0f * dy);
	t1 /= 12.0f;

	Iw[pos] = (t0 + t1) * 0.5f;

	// t derivative
	Iz[pos] = tex2D(texI1, x, y) - tex2D(texI0, x, y);
}

///CUDA CALL FUNCTIONS ***********************************************************
void StereoTgv::ComputeDerivativesFisheyeMasked(float *I0, float *I1, float2 *vector, float* mask,
	int w, int h, int s, float *Iw, float *Iz)
{
	dim3 threads(BlockWidth, BlockHeight);
	dim3 blocks(iDivUp(w, threads.x), iDivUp(h, threads.y));

	// mirror if a coordinate value is out-of-range
	texI0.addressMode[0] = cudaAddressModeMirror;
	texI0.addressMode[1] = cudaAddressModeMirror;
	texI0.filterMode = cudaFilterModeLinear;
	texI0.normalized = true;

	texI1.addressMode[0] = cudaAddressModeMirror;
	texI1.addressMode[1] = cudaAddressModeMirror;
	texI1.filterMode = cudaFilterModeLinear;
	texI1.normalized = true;

	//cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();

	cudaBindTexture2D(0, texI0, I0, w, h, s * sizeof(float));
	cudaBindTexture2D(0, texI1, I1, w, h, s * sizeof(float));

	TgvComputeDerivativesFisheyeMaskedKernel << < blocks, threads >> > (vector, mask, w, h, s, Iw, Iz);
}
