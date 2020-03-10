#include "stereo.h"

__global__ 
void SolveSmoothDualTVGlobalKernel (float* duhat, float* dvhat,
	float* pu1, float* pu2,
	float* pv1, float* pv2,
	int width, int height, int stride,
	float tau, float theta,
	float *pu1s, float *pu2s,
	float *pv1s, float* pv2s)
{
	const int ix = threadIdx.x + blockIdx.x * blockDim.x;
	const int iy = threadIdx.y + blockIdx.y * blockDim.y;

	const int pos = ix + iy * stride;

	if (ix >= width || iy >= height) return;
	
//	int left = (ix - 1) + iy * stride;
	int right = (ix + 1) + iy * stride;
	int down = ix + (iy - 1) * stride;
	int up = ix + (iy + 1) * stride;

	//solve derivatives of duhat and dvhat
	float dux, duy, dvx, dvy;
	if ((ix + 1) >= width) {
		//dux = duhat[pos] - duhat[left];
		//dvx = dvhat[pos] - dvhat[left];
		dux = 0;
		dvx = 0;
	}
	else {
		dux = duhat[right] - duhat[pos];
		dvx = dvhat[right] - dvhat[pos];
	}
	if ((iy + 1) >= height) {
		//duy = duhat[pos] - duhat[down];
		//dvy = dvhat[pos] - dvhat[down];
		duy = 0;
		dvy = 0;
	}
	else {
		duy = duhat[up] - duhat[pos];
		dvy = dvhat[up] - dvhat[pos];
	}
	float magdu = sqrt(dux*dux + duy*duy);
	float magdv = sqrt(dvx*dvx + dvy*dvy);
	float fac = tau / theta;

	float pu1sub = pu1[pos];
	float pu2sub = pu2[pos];
	float pv1sub = pv1[pos];
	float pv2sub = pv2[pos];

	for (int k = 0; k < 1; k++) {
		pu1sub = (pu1sub + fac*dux) / (1 + fac*magdu);
		pu2sub = (pu2sub + fac*duy) / (1 + fac*magdu);
		pv1sub = (pv1sub + fac*dvx) / (1 + fac*magdv);
		pv2sub = (pv2sub + fac*dvy) / (1 + fac*magdv);
	}
	pu1s[pos] = pu1sub;
	pu2s[pos] = pu2sub;
	pv1s[pos] = pv1sub;
	pv2s[pos] = pv2sub;
}


void Stereo::SolveSmoothDualTVGlobal(float *duhat, float *dvhat, 
	float *pu1, float *pu2, float *pv1, float *pv2,
	int w, int h, int s,
	float tau, float theta,
	float *pu1s, float*pu2s,
	float *pv1s, float *pv2s
	)
{
	dim3 threads(BlockWidth, BlockHeight);
	dim3 blocks(iDivUp(w, threads.x), iDivUp(h, threads.y));

	SolveSmoothDualTVGlobalKernel <<< blocks, threads >>> (duhat, dvhat, 
		pu1, pu2, pv1, pv2, 
		w, h, s, 
		tau, theta,
		pu1s, pu2s, pv1s, pv2s);
}


// ***************************
// Dual TV Stereo
// ***************************
__global__
void SolveSmoothDualTVGlobalStereoKernel(float* dwhat,
	float* pw1, float* pw2,
	int width, int height, int stride,
	float tau, float theta,
	float *pw1s, float *pw2s)
{
	const int ix = threadIdx.x + blockIdx.x * blockDim.x;
	const int iy = threadIdx.y + blockIdx.y * blockDim.y;

//	const int pos = ix + iy * stride;
	float desiredRadius = (float)width / 2.20f;
	float halfWidth = (float)width / 2.0f;
	float halfHeight = (float)height / 2.0f;
	float radius = sqrtf((iy - halfHeight) * (iy - halfHeight) + (ix - halfWidth) * (ix - halfWidth));

	if ((iy < height) && (ix < width))
	{
		int pos = ix + iy * stride;// current pixel index
		if (radius >= halfWidth)
		{
			pw1s[pos] = 0.0f;
			pw2s[pos] = 0.0f;
		}
		else
		{
			//int left = (ix - 1) + iy * stride;
			int right = (ix + 1) + iy * stride;
			//int down = ix + (iy - 1) * stride;
			int up = ix + (iy + 1) * stride;

			//solve derivatives of duhat and dvhat
			float dwx, dwy;
			if ((ix + 1) >= width) {
				//dux = duhat[pos] - duhat[left];
				//dvx = dvhat[pos] - dvhat[left];
				dwx = 0;
			}
			else {
				dwx = dwhat[right] - dwhat[pos];
			}
			if ((iy + 1) >= height) {
				//duy = duhat[pos] - duhat[down];
				//dvy = dvhat[pos] - dvhat[down];
				dwy = 0;
			}
			else {
				dwy = dwhat[up] - dwhat[pos];
			}
			float magdu = sqrt(dwx*dwx + dwy * dwy);
			float fac = tau / theta;

			float pw1sub = pw1[pos];
			float pw2sub = pw2[pos];

			for (int k = 0; k < 1; k++) {
				pw1sub = (pw1sub + fac * dwx) / (1 + fac * magdu);
				pw2sub = (pw2sub + fac * dwy) / (1 + fac * magdu);
			}
			pw1s[pos] = pw1sub;
			pw2s[pos] = pw2sub;
		}
	}
}


void Stereo::SolveSmoothDualTVGlobalStereo(float *duhat,
	float *pw1, float *pw2,
	int w, int h, int s,
	float tau, float theta,
	float *pw1s, float*pw2s
)
{
	dim3 threads(BlockWidth, BlockHeight);
	dim3 blocks(iDivUp(w, threads.x), iDivUp(h, threads.y));

	SolveSmoothDualTVGlobalStereoKernel << < blocks, threads >> > (duhat,
		pw1, pw2,
		w, h, s,
		tau, theta,
		pw1s, pw2s);
}
