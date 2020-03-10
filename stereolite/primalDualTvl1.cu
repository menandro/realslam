#include "stereolite.h"

// Solve Problem1A - Thresholding
__global__ void LiteThresholdingL1MaskedKernel(float* mask, float* u, float* u_, float* Iu, float* Iz, float lambda, float theta,
	int width, int height, int stride)
{
	int iy = blockIdx.y * blockDim.y + threadIdx.y;        // current row 
	int ix = blockIdx.x * blockDim.x + threadIdx.x;        // current column 

	if ((iy >= height) && (ix >= width)) return;
	int pos = ix + iy * stride;
	if (mask[pos] == 0.0f) return;

	// Thresholding
	float u_pos = u_[pos];
	float dun = (u[pos] - u_pos);
	float Ius = Iu[pos];
	float rho = Ius * dun + Iz[pos];

	float upper = lambda * theta * (Ius * Ius);
	float lower = -lambda * theta *(Ius * Ius);
	float du;

	if ((rho <= upper) && (rho >= lower)) {
		if (Ius == 0) {
			du = dun;
		}
		else {
			du = dun - rho / Ius;
		}
	}
	else if (rho < lower) {
		du = dun + lambda * theta *Ius;
	}
	else if (rho > upper) {
		du = dun - lambda * theta *Ius;
	}

	u[pos] = u_pos + du;
}

void StereoLite::ThresholdingL1Masked(float* mask, float* u, float* u_, float* Iu, float* Iz, float lambda, float theta, 
	int w, int h, int s)
{
	dim3 threads(BlockWidth, BlockHeight);
	dim3 blocks(iDivUp(w, threads.x), iDivUp(h, threads.y));

	LiteThresholdingL1MaskedKernel << < blocks, threads >> > (mask, u, u_, Iu, Iz, lambda, theta,
		w, h, s);
}


__global__ void LiteThresholdingL1LambdaMaskedKernel(float* mask, float* u, float* u_, float* Iu, float* Iz, float lambda, float* lambdaMask, float theta,
	int width, int height, int stride, float* us)
{
	int iy = blockIdx.y * blockDim.y + threadIdx.y;        // current row 
	int ix = blockIdx.x * blockDim.x + threadIdx.x;        // current column 

	if ((iy >= height) && (ix >= width)) return;
	int pos = ix + iy * stride;
	if (mask[pos] == 0.0f) return;

	if (lambdaMask[pos] == 0.0f) {
		us[pos] = u_[pos];
		return;
	}

	// Thresholding
	float u_pos = u_[pos];
	float dun = (u[pos] - u_pos);
	float Ius = Iu[pos];
	float rho = Ius * dun + Iz[pos];

	float upper = lambda * theta * (Ius * Ius);
	float lower = -lambda * theta *(Ius * Ius);
	float du;

	if ((rho <= upper) && (rho >= lower)) {
		if (Ius == 0) {
			du = dun;
		}
		else {
			du = dun - rho / Ius;
		}
	}
	else if (rho < lower) {
		du = dun + lambda * theta *Ius;
	}
	else if (rho > upper) {
		du = dun - lambda * theta *Ius;
	}

	us[pos] = u_pos + du;
}

void StereoLite::ThresholdingL1LambdaMasked(float* mask, float* u, float* u_, float* Iu, float* Iz, float lambda, float* lambdaMask, float theta,
	float* us, int w, int h, int s)
{
	dim3 threads(BlockWidth, BlockHeight);
	dim3 blocks(iDivUp(w, threads.x), iDivUp(h, threads.y));

	LiteThresholdingL1LambdaMaskedKernel << < blocks, threads >> > (mask, u, u_, Iu, Iz, lambda, lambdaMask, theta,
		w, h, s, us);
}

__global__ void LiteSparsePriorL1Kernel(float* mask, float* u, float* u_, float* usparse, float* Iu, float* Iz, float lambda, float l2lambda, 
	float* lambdaMask, float theta,
	int width, int height, int stride, float* us)
{
	int iy = blockIdx.y * blockDim.y + threadIdx.y;        // current row 
	int ix = blockIdx.x * blockDim.x + threadIdx.x;        // current column 

	if ((iy >= height) && (ix >= width)) return;
	int pos = ix + iy * stride;
	if (mask[pos] == 0.0f) return;

	// Thresholding
	float u_pos = u_[pos];
	float dun = (u[pos] - u_pos);
	float Ius = Iu[pos];
	float rho = Ius * dun + Iz[pos];

	float upper = lambda * theta * (Ius * Ius);
	float lower = -lambda * theta *(Ius * Ius);
	float du;

	if ((rho <= upper) && (rho >= lower)) {
		if (Ius == 0) {
			du = dun;
		}
		else {
			du = dun - rho / Ius;
		}
	}
	else if (rho < lower) {
		du = dun + lambda * theta *Ius;
	}
	else if (rho > upper) {
		du = dun - lambda * theta *Ius;
	}

	float ul1 = u_pos + du;
	float usparsepos = usparse[pos];
	/*if (abs(ul1 - usparsepos) > 5.0f) {
		usparsepos = ul1;
	}*/

	if (lambdaMask[pos] == 0.0f) {
		us[pos] = ul1;
	}
	else {
		us[pos] = (ul1 + l2lambda * usparsepos) / (1.0f + l2lambda);
	}
}

void StereoLite::SparsePriorL1(float* fisheyeMask, float* u, float* u_, float * usparse, float* Iu, float* Iz, 
	float lambda, float l2lambda, float* lambdaMask, float theta,
	float* us, int w, int h, int s)
{
	dim3 threads(BlockWidth, BlockHeight);
	dim3 blocks(iDivUp(w, threads.x), iDivUp(h, threads.y));

	LiteSparsePriorL1Kernel << < blocks, threads >> > (fisheyeMask, u, u_, usparse, Iu, Iz, lambda, l2lambda, 
		lambdaMask, theta,
		w, h, s, us);
}


__global__ void LiteL2Kernel (float* mask, float* u, float* u_, float l2lambda, float* lambdaMask, 
	int width, int height, int stride, float* us)
{
	int iy = blockIdx.y * blockDim.y + threadIdx.y;        // current row 
	int ix = blockIdx.x * blockDim.x + threadIdx.x;        // current column 

	if ((iy >= height) && (ix >= width)) return;
	int pos = ix + iy * stride;
	if (mask[pos] == 0.0f) return;

	if (lambdaMask[pos] == 0.0f) {
		us[pos] = u_[pos];
	}
	else {
		us[pos] = (u_[pos] + l2lambda * u[pos])/(1.0f + l2lambda);
	}
}

void StereoLite::SimpleL2(float* mask, float* u, float* u_, float l2lambda, float* lambdaMask,
	float* us, int w, int h, int s)
{
	dim3 threads(BlockWidth, BlockHeight);
	dim3 blocks(iDivUp(w, threads.x), iDivUp(h, threads.y));

	LiteL2Kernel << < blocks, threads >> > (mask, u, u_, l2lambda, lambdaMask, w, h, s, us);
}

// Solve Problem1B
__global__ void LiteSolveProblem1bMaskedKernel(float* mask, float* u, float2 *p, float theta,
	int width, int height, int stride, float* us)
{
	int iy = blockIdx.y * blockDim.y + threadIdx.y;        // current row 
	int ix = blockIdx.x * blockDim.x + threadIdx.x;        // current column 

	if ((iy >= height) && (ix >= width)) return;
	int pos = ix + iy * stride;
	if (mask[pos] == 0.0f) return;

	int left = (ix - 1) + iy * stride;
	int up = ix + (iy - 1) * stride;

	float maskLeft, maskUp;
	if (ix - 1 < 0) {
		maskLeft = 0.0f;
	}
	else maskLeft = mask[left];

	if (iy - 1 < 0) {
		maskUp = 0.0f;
	}
	else maskUp = mask[up];

	float divp;
	float2 ppos = p[pos];
	if (maskLeft == 0.0f) {
		if (maskUp == 0.0f) {
			//divp = ppos.x + ppos.y;
			divp = 0.0f;
		}
		else {
			//divp = ppos.x + ppos.y - p[up].y;
			divp = ppos.y - p[up].y;
		}
	}
	else {
		if (maskUp == 0.0f) {
			//divp = ppos.x - p[left].x + ppos.y;
			divp = ppos.x - p[left].x;
		}
		else {
			divp = ppos.x - p[left].x + ppos.y - p[up].y;
		}
	}
	us[pos] = u[pos] + theta * divp;
}

void StereoLite::SolveProblem1bMasked(float* mask, float* u, float2 *p, float theta,
	float* us, int w, int h, int s)
{
	dim3 threads(BlockWidth, BlockHeight);
	dim3 blocks(iDivUp(w, threads.x), iDivUp(h, threads.y));

	LiteSolveProblem1bMaskedKernel << < blocks, threads >> > (mask, u, p, theta,
		w, h, s, us);
}

// Solve Problem 2
__global__ void LiteSolveProblem2MaskedKernel(float* mask, float* u, float2 *p, float theta, float tau,
	int width, int height, int stride, float2* ps)
{
	int iy = blockIdx.y * blockDim.y + threadIdx.y;        // current row 
	int ix = blockIdx.x * blockDim.x + threadIdx.x;        // current column 

	if ((iy >= height) && (ix >= width)) return;
	int pos = ix + iy * stride;
	if (mask[pos] == 0.0f) return;

	int right = (ix + 1) + iy * stride;
	int down = ix + (iy + 1) * stride;
	int left = (ix - 1) + iy * stride;
	int up = ix + (iy - 1) * stride;
	float maskRight, maskDown;
	if (ix + 1 >= width) {
		maskRight = 0.0f;
	}
	else maskRight = mask[right];
	if (iy + 1 >= height) {
		maskDown = 0.0f;
	}
	else maskDown = mask[down];

	float dux, duy;
	if (maskRight == 0.0f) {
		dux = 0;
	}
	else {
		dux = u[right] - u[pos];
	}
	if (maskDown == 0.0f) {
		duy = 0;
	}
	else {
		duy = u[down] - u[pos];
	}

	float magdu = sqrt(dux * dux + duy * duy);
	float fac = tau / theta;

	float2 psub = p[pos];

	ps[pos].x = (psub.x + fac * dux) / (1 + fac * magdu);
	ps[pos].y = (psub.y + fac * duy) / (1 + fac * magdu);
}

void StereoLite::SolveProblem2Masked(float* mask, float* u, float2 *p, float theta, float tau,
	float2* ps, int w, int h, int s)
{
	dim3 threads(BlockWidth, BlockHeight);
	dim3 blocks(iDivUp(w, threads.x), iDivUp(h, threads.y));

	LiteSolveProblem2MaskedKernel << < blocks, threads >> > (mask, u, p, theta, tau, w, h, s, ps);
}