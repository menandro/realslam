#include "stereo.h"

__global__
void SolveDataL1Kernel(const float *duhat0, const float *dvhat0,
	const float *pu1, const float *pu2,
	const float *pv1, const float *pv2,
	const float *Ix, const float *Iy, const float *It,
	int width, int height, int stride,
	float lambda, float theta,
	float *duhat1, float *dvhat1)
{
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;        // current row 

	float dix, diy, dit, duhat, dvhat, du, dv;

	if ((iy < height) && (ix < width))
	{
		int pos = ix + iy * stride;      // current pixel index
		dix = Ix[pos];
		diy = Iy[pos];
		dit = It[pos];
		float duhat = duhat0[pos];
		float dvhat = dvhat0[pos];
		
		//problem 1a
		float rho = (dix*duhat + diy*dvhat + dit);
		float upper = lambda*theta*(dix*dix + diy*diy);
		float lower = -lambda*theta*(dix*dix + diy*diy);;

		if ((rho <= upper) && (rho >= lower)) {
			float magi = dix*dix + diy*diy;
			if (magi != 0) {
				du = duhat - rho*dix / magi;
				dv = dvhat - rho*diy / magi;
			}
			else {
				du = duhat;
				dv = dvhat;
			}
			
		}
		else if (rho < lower) {
			du = duhat + lambda*theta*dix;
			dv = dvhat + lambda*theta*diy;
		}
		else if (rho > upper) {
			du = duhat - lambda*theta*dix;
			dv = dvhat - lambda*theta*diy;
		}

		//problem 1b
		float divpu, divpv;
		int left = (ix - 1) + iy * stride;
		int right = (ix + 1) + iy * stride;
		int down = ix + (iy - 1) * stride;
		int up = ix + (iy + 1) * stride;

		if ((ix - 1) < 0) {
			if ((iy - 1) < 0) {
				//divpu = pu1[right] - pu1[pos] + pu2[up] - pu2[pos];
				//divpv = pv1[right] - pv1[pos] + pv2[up] - pv2[pos];
				divpu = pu1[pos] + pu2[pos];
				divpv = pv1[pos] + pv2[pos];
			}
			else {
				//divpu = pu1[right] - pu1[pos] + pu2[pos] - pu2[down];
				//divpv = pv1[right] - pv1[pos] + pv2[pos] - pv2[down];
				divpu = pu1[pos] + pu2[pos] - pu2[down];
				divpv = pv1[pos] + pv2[pos] - pv2[down];
			}
		}
		else {
			if ((iy - 1) < 0) {
				//divpu = pu1[pos] - pu1[left] + pu2[up] - pu2[pos];
				//divpv = pv1[pos] - pv1[left] + pv2[up] - pv2[pos];
				divpu = pu1[pos] - pu1[left] + pu2[pos];
				divpv = pv1[pos] - pv1[left] + pv2[pos];
			}
			else {
				divpu = pu1[pos] - pu1[left] + pu2[pos] - pu2[down];
				divpv = pv1[pos] - pv1[left] + pv2[pos] - pv2[down];
			}
		}

		duhat1[pos] = du + theta*divpu;
		dvhat1[pos] = dv + theta*divpv;
	}

}


void Stereo::SolveDataL1(const float *duhat0, const float *dvhat0,
	const float *pu1, const float *pu2,
	const float *pv1, const float *pv2,
	const float *Ix, const float *Iy, const float *Iz,
	int w, int h, int s,
	float lambda, float theta,
	float *duhat1, float *dvhat1)
{
	// CTA size
	dim3 threads(BlockWidth, BlockHeight);
	// grid size
	dim3 blocks(iDivUp(w, threads.x), iDivUp(h, threads.y));

	SolveDataL1Kernel <<< blocks, threads >>> (duhat0, dvhat0,
		pu1, pu2,
		pv1, pv2,
Ix, Iy, Iz,
w, h, s,
lambda, theta,
duhat1, dvhat1);
}


// *******************************
// Solve Data-L1 Stereo
// *******************************

__global__
void SolveDataL1StereoKernel(const float *dwhat0,
	const float *pw1, const float *pw2,
	const float *Iw, const float *It,
	int width, int height, int stride,
	float lambda, float theta,
	float *dwhat1)
{
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;        // current row 

	float diw, dit, dwhat, dw;

	float desiredRadius = (float)width / 2.20f;
	float halfWidth = (float)width / 2.0f;
	float halfHeight = (float)height / 2.0f;
	float radius = sqrtf((iy - halfHeight) * (iy - halfHeight) + (ix - halfWidth) * (ix - halfWidth));

	if ((iy < height) && (ix < width))
	{
		int pos = ix + iy * stride;// current pixel index
		if (radius >= halfWidth)
		{
			dwhat1[pos] = 0.0f;
		}
		else
		{
			diw = Iw[pos];
			dit = It[pos];
			float dwhat = dwhat0[pos];

			//problem 1a
			float magi = (diw * diw);
			float rho = (diw*dwhat + dit);
			//float upper = lambda * theta*(diw * diw);
			//float lower = -lambda * theta*(diw * diw);
			float upper = lambda * theta*(magi);
			float lower = -lambda * theta*(magi);

			if ((rho <= upper) && (rho >= lower)) {
				if (diw != 0) {
					dw = dwhat - rho / diw;
				}
				else {
					dw = dwhat;
				}

			}
			else if (rho < lower) {
				dw = dwhat + lambda * theta*diw;
			}
			else if (rho > upper) {
				dw = dwhat - lambda * theta*diw;
			}

			/*if (dw > 0.5f) {
				dw = 0.5f;
			}*/
			/*else if (dw < 0.0f) {
				dw = 0.0f;
			}*/

			//problem 1b
			float divpu, divpv;
			int left = (ix - 1) + iy * stride;
			int right = (ix + 1) + iy * stride;
			int down = ix + (iy - 1) * stride;
			int up = ix + (iy + 1) * stride;

			if ((ix - 1) < 0) {
				if ((iy - 1) < 0) {
					//divpu = pu1[right] - pu1[pos] + pu2[up] - pu2[pos];
					//divpv = pv1[right] - pv1[pos] + pv2[up] - pv2[pos];
					divpu = pw1[pos] + pw2[pos];
				}
				else {
					//divpu = pu1[right] - pu1[pos] + pu2[pos] - pu2[down];
					//divpv = pv1[right] - pv1[pos] + pv2[pos] - pv2[down];
					divpu = pw1[pos] + pw2[pos] - pw2[down];
				}
			}
			else {
				if ((iy - 1) < 0) {
					//divpu = pu1[pos] - pu1[left] + pu2[up] - pu2[pos];
					//divpv = pv1[pos] - pv1[left] + pv2[up] - pv2[pos];
					divpu = pw1[pos] - pw1[left] + pw2[pos];
				}
				else {
					divpu = pw1[pos] - pw1[left] + pw2[pos] - pw2[down];
				}
			}

			float dwval = dw + theta * divpu;
			dwhat1[pos] = dwval;
			/*if (dwval < 0) {
				dwhat1[pos] = 0.0f;
			}
			else if (dwval > 0.5f) {
				dwhat1[pos] = 0.5f;
			}
			else {
				dwhat1[pos] = dwval;
			}*/
		}
		
		//dwhat1[pos] = dw + theta * divpu;
	}

}


void Stereo::SolveDataL1Stereo(const float *dwhat0,
	const float *pw1, const float *pw2,
	const float *Iw, const float *Iz,
	int w, int h, int s,
	float lambda, float theta,
	float *dwhat1)
{
	// CTA size
	dim3 threads(BlockWidth, BlockHeight);
	// grid size
	dim3 blocks(iDivUp(w, threads.x), iDivUp(h, threads.y));

	SolveDataL1StereoKernel << < blocks, threads >> > (dwhat0,
		pw1, pw2,
		Iw, Iz,
		w, h, s,
		lambda, theta,
		dwhat1);
}