#include "upsampling.h"

// Textures
texture<float, 2, cudaReadModeElementType> gray_img;
texture<float, 2, cudaReadModeElementType> imgToFilter;

// Calculate weight
__global__ void CalcWeightKernel(float *input, float* weight, float lambda_tgvl2, int width, int height, int stride) {
	int iy = blockIdx.y * blockDim.y + threadIdx.y;        // current row 
	int ix = blockIdx.x * blockDim.x + threadIdx.x;        // current column 

	if ((iy < height) && (ix < width))
	{
		int pos = ix + iy * stride;

		if (input[pos] > 0) {
			weight[pos] = 1.0f;
		}
		else {
			weight[pos] = 0.0f;
		}
		weight[pos] = weight[pos] * lambda_tgvl2;
	}
}

void lup::Upsampling::CalcWeight(float *input, float *weight, float lambda_tgvl2) {
	dim3 threads(BlockWidth, BlockHeight);
	dim3 blocks(iDivUp(width, threads.x), iDivUp(height, threads.y));
	CalcWeightKernel << < blocks, threads >> > (input, weight, lambda_tgvl2, width, height, stride);
}

// Calculate anisotropic diffusion tensor
__global__ void CalcTensorKernel(float* gray, float beta, float gamma, int size_grad,
	float* atensor, float* btensor, float* ctensor,
	int width, int height, int stride)
{
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
		/*t0 = tex2D(gray_img, x - 2.0f * dx, y);
		t0 -= tex2D(gray_img, x - 1.0f * dx, y) * 8.0f;
		t0 += tex2D(gray_img, x + 1.0f * dx, y) * 8.0f;
		t0 -= tex2D(gray_img, x + 2.0f * dx, y);
		t0 /= 12.0f;*/
		t0 = tex2D(gray_img, x + 1.0f * dx, y);
		t0 -= tex2D(gray_img, x, y);
		t0 = tex2D(gray_img, x + 1.0f * dx, y + 1.0f * dy);
		t0 -= tex2D(gray_img, x, y + 1.0f * dy);
		grad.x = t0;

		// y derivative
		/*t0 = tex2D(gray_img, x, y - 2.0f * dy);
		t0 -= tex2D(gray_img, x, y - 1.0f * dy) * 8.0f;
		t0 += tex2D(gray_img, x, y + 1.0f * dy) * 8.0f;
		t0 -= tex2D(gray_img, x, y + 2.0f * dy);
		t0 /= 12.0f;*/
		t0 = tex2D(gray_img, x, y + 1.0f * dy);
		t0 -= tex2D(gray_img, x, y);
		t0 = tex2D(gray_img, x + 1.0f * dx, y + 1.0f * dy);
		t0 -= tex2D(gray_img, x + 1.0f * dx, y);
		grad.y = t0;

		float min_n_length = 1e-8f;
		float min_tensor_val = 1e-8f;

		float abs_img = sqrtf(grad.x*grad.x + grad.y*grad.y);
		float norm_n = abs_img;

		float2 n_normed;
		n_normed.x = grad.x / norm_n;
		n_normed.y = grad.y / norm_n;

		if (norm_n < min_n_length) {
			n_normed.x = 1.0f;
			n_normed.y = 0.0f;
		}

		float2 nT_normed;
		nT_normed.x = n_normed.y;
		nT_normed.y = -n_normed.x;

		float wtensor;
		if (expf(-beta * powf(abs_img, gamma)) > min_tensor_val) {
			wtensor = expf(-beta * powf(abs_img, gamma));
		}
		else wtensor = min_tensor_val;

		float a = wtensor * n_normed.x * n_normed.x + nT_normed.x * nT_normed.x;
		float c = wtensor * n_normed.x * n_normed.y + nT_normed.x * nT_normed.y;
		float b = wtensor * n_normed.y * n_normed.y + nT_normed.y * nT_normed.y;
		atensor[pos] = a;
		btensor[pos] = b;
		ctensor[pos] = c;

	}
}


void lup::Upsampling::CalcTensor(float* gray, float beta, float gamma, int size_grad,
	float* a, float* b, float* c)
{
	dim3 threads(BlockWidth, BlockHeight);
	dim3 blocks(iDivUp(width, threads.x), iDivUp(height, threads.y));

	// mirror if a coordinate value is out-of-range
	gray_img.addressMode[0] = cudaAddressModeMirror;
	gray_img.addressMode[1] = cudaAddressModeMirror;
	gray_img.filterMode = cudaFilterModeLinear;
	gray_img.normalized = true;

	cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();

	cudaBindTexture2D(0, gray_img, gray, width, height, stride * sizeof(float));
	CalcTensorKernel << < blocks, threads >> > (gray, beta, gamma, size_grad,
		a, b, c,
		width, height, stride);
}


// Gaussian Filter
// Calculate anisotropic diffusion tensor
__global__ void GaussianKernel(float* input, float* output,
	int width, int height, int stride)
{
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
		float t0 = (1 / 4.0f)*tex2D(imgToFilter, x, y);
		t0 += (1 / 16.0f)*tex2D(imgToFilter, x - 1.0f * dx, y - 1.0f * dy);
		t0 += (1 / 16.0f)*tex2D(imgToFilter, x - 1.0f * dx, y + 1.0f * dy);
		t0 += (1 / 16.0f)*tex2D(imgToFilter, x + 1.0f * dx, y - 1.0f * dy);
		t0 += (1 / 16.0f)*tex2D(imgToFilter, x + 1.0f * dx, y + 1.0f * dy);
		t0 += (1 / 8.0f)*tex2D(imgToFilter, x - 1.0f * dx, y);
		t0 += (1 / 8.0f)*tex2D(imgToFilter, x + 1.0f * dx, y);
		t0 += (1 / 8.0f)*tex2D(imgToFilter, x, y - 1.0f * dy);
		t0 += (1 / 8.0f)*tex2D(imgToFilter, x, y + 1.0f * dy);

		output[pos] = t0;
	}
}


void lup::Upsampling::Gaussian(float* input, float* output)
{
	dim3 threads(BlockWidth, BlockHeight);
	dim3 blocks(iDivUp(width, threads.x), iDivUp(height, threads.y));

	// mirror if a coordinate value is out-of-range
	imgToFilter.addressMode[0] = cudaAddressModeMirror;
	imgToFilter.addressMode[1] = cudaAddressModeMirror;
	imgToFilter.filterMode = cudaFilterModeLinear;
	imgToFilter.normalized = true;

	cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();

	cudaBindTexture2D(0, imgToFilter, input, width, height, stride * sizeof(float));
	GaussianKernel << < blocks, threads >> > (input, output,
		width, height, stride);
}


// Solve eta_u, eta_v
__global__ void SolveEtaKernel(float* weights, float alpha0, float alpha1,
	float* atensor, float *btensor, float* ctensor,
	float* etau, float* etav1, float* etav2,
	int width, int height, int stride)
{
	int iy = blockIdx.y * blockDim.y + threadIdx.y;        // current row 
	int ix = blockIdx.x * blockDim.x + threadIdx.x;        // current column 

	if ((iy < height) && (ix < width))
	{
		int pos = ix + iy * stride;
		float a = atensor[pos];
		float b = btensor[pos];
		float c = ctensor[pos];

		etau[pos] = (a*a + b * b + 2 * c*c + (a + c)*(a + c) + (b + c)*(b + c)) * (alpha1 * alpha1) + 0 * weights[pos] * weights[pos];
		etav1[pos] = (alpha1 * alpha1)*(b * b + c * c) + 4 * alpha0 * alpha0;
		etav2[pos] = (alpha1 * alpha1)*(a * a + c * c) + 4 * alpha0 * alpha0;
	}
}

void lup::Upsampling::SolveEta(float* weights, float alpha0, float alpha1,
	float* a, float *b, float* c,
	float* etau, float* etav1, float* etav2)
{
	dim3 threads(BlockWidth, BlockHeight);
	dim3 blocks(iDivUp(width, threads.x), iDivUp(height, threads.y));
	SolveEtaKernel << < blocks, threads >> > (weights, alpha0, alpha1,
		a, b, c,
		etau, etav1, etav2,
		width, height, stride);
}


// Multiply two matrices
__global__ void MultKernel(float* input0, float*input1, float* output, int width, int height, int stride) {
	int iy = blockIdx.y * blockDim.y + threadIdx.y;        // current row 
	int ix = blockIdx.x * blockDim.x + threadIdx.x;        // current column 

	if ((iy < height) && (ix < width))
	{
		int pos = ix + iy * stride;
		output[pos] = input0[pos] * input1[pos];
	}
}

__global__ void MultKernel(float* input0, float scale, float* output, int width, int height, int stride) {
	int iy = blockIdx.y * blockDim.y + threadIdx.y;        // current row 
	int ix = blockIdx.x * blockDim.x + threadIdx.x;        // current column 

	if ((iy < height) && (ix < width))
	{
		int pos = ix + iy * stride;
		output[pos] = input0[pos] * scale;
	}
}

void lup::Upsampling::Mult(float* input0, float* input1, float *output) {
	dim3 threads(BlockWidth, BlockHeight);
	dim3 blocks(iDivUp(width, threads.x), iDivUp(height, threads.y));
	MultKernel << < blocks, threads >> > (input0, input1, output, width, height, stride);
}

void lup::Upsampling::Mult(float* input0, float scale, float *output) {
	dim3 threads(BlockWidth, BlockHeight);
	dim3 blocks(iDivUp(width, threads.x), iDivUp(height, threads.y));
	MultKernel << < blocks, threads >> > (input0, scale, output, width, height, stride);
}


__global__ void NormalizeKernel(float* input, float min, float max, float* output, int width, int height, int stride) {
	int iy = blockIdx.y * blockDim.y + threadIdx.y;        // current row 
	int ix = blockIdx.x * blockDim.x + threadIdx.x;        // current column 

	if ((iy < height) && (ix < width))
	{
		int pos = ix + iy * stride;
		output[pos] = (input[pos] - min) / (max - min);
	}
}

void lup::Upsampling::Normalize(float* input, float min, float max, float *output) {
	dim3 threads(BlockWidth, BlockHeight);
	dim3 blocks(iDivUp(width, threads.x), iDivUp(height, threads.y));
	NormalizeKernel << < blocks, threads >> > (input, min, max, output, width, height, stride);
}

__global__ void NormalizeClipKernel(float* input, float min, float max, float* output, int width, int height, int stride) {
	int iy = blockIdx.y * blockDim.y + threadIdx.y;        // current row 
	int ix = blockIdx.x * blockDim.x + threadIdx.x;        // current column 

	if ((iy < height) && (ix < width))
	{
		int pos = ix + iy * stride;
		output[pos] = (input[pos] - min) / (max - min);
		if (output[pos] < 0.0f) {
			output[pos] = 0.0f;
		}
		if (output[pos] > 1.0f) {
			output[pos] = 1.0f;
		}
	}
}

void lup::Upsampling::NormalizeClip(float* input, float min, float max, float *output) {
	dim3 threads(BlockWidth, BlockHeight);
	dim3 blocks(iDivUp(width, threads.x), iDivUp(height, threads.y));
	NormalizeClipKernel << < blocks, threads >> > (input, min, max, output, width, height, stride);
}

__global__ void DenormalizeClipKernel(float* input, float min, float max, float* output, int width, int height, int stride) {
	int iy = blockIdx.y * blockDim.y + threadIdx.y;        // current row 
	int ix = blockIdx.x * blockDim.x + threadIdx.x;        // current column 

	if ((iy < height) && (ix < width))
	{
		int pos = ix + iy * stride;
		output[pos] = input[pos] * (max - min) + min;
	}
}

void lup::Upsampling::DenormalizeClip(float* input, float min, float max, float *output) {
	dim3 threads(BlockWidth, BlockHeight);
	dim3 blocks(iDivUp(width, threads.x), iDivUp(height, threads.y));
	DenormalizeClipKernel << < blocks, threads >> > (input, min, max, output, width, height, stride);
}


// Update Dual Variables (p, q)
__global__ void UpdateDualVariablesTGVKernel(float* u_, float2 *v_, float alpha0, float alpha1, float sigma,
	float eta_p, float eta_q,
	float* a, float* b, float*c,
	float4* grad_v, float2* p, float4* q,
	int width, int height, int stride)
{
	int iy = blockIdx.y * blockDim.y + threadIdx.y;        // current row 
	int ix = blockIdx.x * blockDim.x + threadIdx.x;        // current column 

	if ((iy < height) && (ix < width))
	{
		int pos = ix + iy * stride;
		int right = (ix + 1) + iy * stride;
		int down = ix + (iy + 1) * stride;
		int left = (ix - 1) + iy * stride;
		int up = ix + (iy - 1) * stride;

		//u_x = dxp(u_) - v_(:, : , 1);
		float u_x, u_y;
		if ((ix + 1) < width) u_x = u_[right] - u_[pos] - v_[pos].x;
		else u_x = u_[pos] - u_[left] - v_[pos].x;
		//u_y = dyp(u_) - v_(:, : , 2);
		if ((iy + 1) < height) u_y = u_[down] - u_[pos] - v_[pos].y;
		else u_y = u_[pos] - u_[up] - v_[pos].y;

		//du_tensor_x = a.*u_x + c.*u_y;
		float du_tensor_x = a[pos] * u_x + c[pos] * u_y;
		//du_tensor_y = c.*u_x + b.*u_y;
		float du_tensor_y = c[pos] * u_x + b[pos] * u_y;

		//p(:, : , 1) = p(:, : , 1) + alpha1*sigma / eta_p.*du_tensor_x;
		p[pos].x = p[pos].x + (alpha1*sigma / eta_p) * du_tensor_x;
		//p(:, : , 2) = p(:, : , 2) + alpha1*sigma / eta_p.*du_tensor_y;
		p[pos].y = p[pos].y + (alpha1*sigma / eta_p) * du_tensor_y;

		//projection
		//reprojection = max(1.0, sqrt(p(:, : , 1). ^ 2 + p(:, : , 2). ^ 2));
		float reprojection = sqrtf(p[pos].x * p[pos].x + p[pos].y * p[pos].y);
		if (reprojection < 1.0f) {
			reprojection = 1.0f;
		}
		//p(:, : , 1) = p(:, : , 1). / reprojection;
		p[pos].x = p[pos].x / reprojection;
		//p(:, : , 2) = p(:, : , 2). / reprojection;
		p[pos].y = p[pos].y / reprojection;

		//grad_v(:, : , 1) = dxp(v_(:, : , 1));
		if ((ix + 1) < width) grad_v[pos].x = v_[right].x - v_[pos].x;
		else grad_v[pos].x = v_[pos].x - v_[left].x;

		//grad_v(:, : , 2) = dyp(v_(:, : , 2));
		if ((iy + 1) < height) grad_v[pos].y = v_[down].y - v_[pos].y;
		else grad_v[pos].y = v_[pos].y - v_[up].y;

		//grad_v(:, : , 3) = dyp(v_(:, : , 1));
		if ((iy + 1) < height) grad_v[pos].z = v_[down].x - v_[pos].x;
		else grad_v[pos].z = v_[pos].x - v_[up].x;

		//grad_v(:, : , 4) = dxp(v_(:, : , 2));
		if ((ix + 1) < width) grad_v[pos].w = v_[right].y - v_[pos].y;
		else grad_v[pos].w = v_[pos].y - v_[left].y;

		//q = q + alpha0*sigma / eta_q.*grad_v;
		float ase = alpha0 * sigma / eta_q;
		q[pos].x = q[pos].x + ase * grad_v[pos].x;
		q[pos].y = q[pos].y + ase * grad_v[pos].y;
		q[pos].z = q[pos].z + ase * grad_v[pos].z;
		q[pos].w = q[pos].w + ase * grad_v[pos].w;

		//reproject = max(1.0, sqrt(q(:, : , 1). ^ 2 + q(:, : , 2). ^ 2 + q(:, : , 3). ^ 2 + q(:, : , 4). ^ 2));
		float reproject = sqrtf(q[pos].x * q[pos].x + q[pos].y * q[pos].y + q[pos].z * q[pos].z + q[pos].w * q[pos].w);
		if (reproject < 1.0f) {
			reproject = 1.0f;
		}
		//q(:, : , 1) = q(:, : , 1). / reproject;
		q[pos].x = q[pos].x / reproject;
		//q(:, : , 2) = q(:, : , 2). / reproject;
		q[pos].y = q[pos].y / reproject;
		//q(:, : , 3) = q(:, : , 3). / reproject;
		q[pos].z = q[pos].z / reproject;
		//q(:, : , 4) = q(:, : , 4). / reproject;
		q[pos].w = q[pos].w / reproject;
	}
}


void lup::Upsampling::UpdateDualVariablesTGV(float* u_, float2 *v_, float alpha0, float alpha1, float sigma,
	float eta_p, float eta_q,
	float* a, float* b, float* c,
	float4* grad_v, float2* p, float4* q)
{
	dim3 threads(BlockWidth, BlockHeight);
	dim3 blocks(iDivUp(width, threads.x), iDivUp(height, threads.y));
	UpdateDualVariablesTGVKernel << < blocks, threads >> > (u_, v_, alpha0, alpha1, sigma, eta_p, eta_q,
		a, b, c,
		grad_v, p, q,
		width, height, stride);
}

// Solve Tp
__global__ void SolveTpKernel(float*a, float *b, float*c, float2* p, float2* Tp, int width, int height, int stride) {
	int iy = blockIdx.y * blockDim.y + threadIdx.y;        // current row 
	int ix = blockIdx.x * blockDim.x + threadIdx.x;        // current column 

	if ((iy < height) && (ix < width))
	{
		int pos = ix + iy * stride;

		Tp[pos].x = a[pos] * p[pos].x + c[pos] * p[pos].y;
		Tp[pos].y = c[pos] * p[pos].x + b[pos] * p[pos].y;
	}
}

void lup::Upsampling::SolveTp(float* a, float* b, float* c, float2* p, float2* Tp) {
	dim3 threads(BlockWidth, BlockHeight);
	dim3 blocks(iDivUp(width, threads.x), iDivUp(height, threads.y));
	SolveTpKernel << < blocks, threads >> > (a, b, c, p, Tp, width, height, stride);
}


// Clone
__global__ void CloneKernel(float* dst, float* src, int width, int height, int stride) {
	int iy = blockIdx.y * blockDim.y + threadIdx.y;        // current row 
	int ix = blockIdx.x * blockDim.x + threadIdx.x;        // current column 

	if ((iy < height) && (ix < width))
	{
		int pos = ix + iy * stride;
		dst[pos] = src[pos];
	}
}

__global__ void CloneKernel2(float2* dst, float2* src, int width, int height, int stride) {
	int iy = blockIdx.y * blockDim.y + threadIdx.y;        // current row 
	int ix = blockIdx.x * blockDim.x + threadIdx.x;        // current column 

	if ((iy < height) && (ix < width))
	{
		int pos = ix + iy * stride;
		dst[pos] = src[pos];
	}
}

void lup::Upsampling::Clone(float* dst, float* src) {
	dim3 threads(BlockWidth, BlockHeight);
	dim3 blocks(iDivUp(width, threads.x), iDivUp(height, threads.y));
	CloneKernel << < blocks, threads >> > (dst, src, width, height, stride);
}

void lup::Upsampling::Clone(float2* dst, float2* src) {
	dim3 threads(BlockWidth, BlockHeight);
	dim3 blocks(iDivUp(width, threads.x), iDivUp(height, threads.y));
	CloneKernel2 << < blocks, threads >> > (dst, src, width, height, stride);
}


// Update Primal variables L2 (u, v)
__global__ void UpdatePrimalVariablesL2Kernel(float2* Tp, float* u_, float2* v_, float2* p, float4* q,
	float* a, float* b, float* c,
	float tau, float* eta_u, float* eta_v1, float* eta_v2,
	float alpha0, float alpha1, float* w, float* dw, float mu,
	float* u, float2* v,
	float* u_s, float2* v_s,
	int width, int height, int stride)
{
	int iy = blockIdx.y * blockDim.y + threadIdx.y;        // current row 
	int ix = blockIdx.x * blockDim.x + threadIdx.x;        // current column 

	if ((iy < height) && (ix < width))
	{
		int pos = ix + iy * stride;
		int right = (ix + 1) + iy * stride;
		int down = ix + (iy + 1) * stride;
		int left = (ix - 1) + iy * stride;
		int up = ix + (iy - 1) * stride;

		//div_p = dxm(Tp(:, : , 1)) + dym(Tp(:, : , 2));
		float div_p;
		float dxmTp, dymTp;
		if ((ix - 1) >= 0)
			dxmTp = Tp[pos].x - Tp[left].x;
		else if (ix == width - 1)
			dxmTp = -Tp[left].x;
		else
			dxmTp = Tp[pos].x;

		if ((iy - 1) >= 0)
			dymTp = Tp[pos].y - Tp[up].y;
		else if (iy == height - 1)
			dymTp = -Tp[up].y;
		else
			dymTp = Tp[pos].y;

		div_p = dxmTp + dymTp;

		//tau_eta_u = tau. / eta_u;
		float tau_eta_u = tau / eta_u[pos];

		//u = (u_ + tau_eta_u.*(alpha1.*div_p + dw)). / (1 + tau_eta_u.*w);
		u[pos] = (u_[pos] + tau_eta_u * (alpha1 * div_p + dw[pos])) / (1 + tau_eta_u * w[pos]);

		//qc(:, : , 1) = [q(:, 1 : end - 1, 1), zeros(M, 1)];
		//qc(:, : , 2) = [q(1:end - 1, : , 2); zeros(1, N)];
		//qc(:, : , 3) = [q(1:end - 1, : , 3); zeros(1, N)];
		//qc(:, : , 4) = [q(:, 1 : end - 1, 4), zeros(M, 1)];
		float4 qc;
		if (ix == width - 1) {
			qc.x = 0.0f;
			qc.w = 0.0f;
		}
		else {
			qc.x = q[pos].x;
			qc.w = q[pos].w;
		}
		if (iy == height - 1) {
			qc.y = 0.0f;
			qc.z = 0.0f;
		}
		else {
			qc.y = q[pos].y;
			qc.z = q[pos].z;
		}

		//qw_x = [zeros(M, 1, 1), q(:, 1 : end - 1, 1)];
		//qw_w = [zeros(M, 1, 1), q(:, 1 : end - 1, 4)];
		float qw_x, qw_w;
		if ((ix - 1) >= 0) {
			qw_x = q[left].x;
			qw_w = q[left].w;
		}
		else {
			qw_x = 0.0f;
			qw_w = 0.0f;
		}

		//qn_y = [zeros(1, N, 1); q(1:end - 1, : , 2)];
		//qn_z = [zeros(1, N, 1); q(1:end - 1, : , 3)];
		float qn_y, qn_z;
		if ((iy - 1) >= 0) {
			qn_y = q[up].y;
			qn_z = q[up].z;
		}
		else {
			qn_y = 0.0f;
			qn_z = 0.0f;
		}

		//div_q(:, : , 1) = (qc(:, : , 1) - qw_x) + (qc(:, : , 3) - qn_z);
		//div_q(:, : , 2) = (qc(:, : , 4) - qw_w) + (qc(:, : , 2) - qn_y);
		float2 div_q;
		div_q.x = (qc.x - qw_x) + (qc.z - qn_z);
		div_q.y = (qc.w - qw_w) + (qc.y - qn_y);

		//dq_tensor(:, : , 1) = a.*p(:, : , 1) + c.*p(:, : , 2);
		//dq_tensor(:, : , 2) = c.*p(:, : , 1) + b.*p(:, : , 2);
		float2 dq_tensor;
		dq_tensor.x = a[pos] * p[pos].x + c[pos] * p[pos].y;
		dq_tensor.y = c[pos] * p[pos].x + b[pos] * p[pos].y;

		//v = v_ + tau. / eta_v.*(alpha1.*dq_tensor + alpha0.*div_q);
		v[pos].x = v_[pos].x + (tau / eta_v1[pos]) * (alpha1 * dq_tensor.x + alpha0 * div_q.x);
		v[pos].y = v_[pos].y + (tau / eta_v2[pos]) * (alpha1 * dq_tensor.y + alpha0 * div_q.y);

		// over - relaxation
		//u_ = u + mu.*(u - u_);
		//v_ = v + mu.*(v - v_);
		u_s[pos] = u[pos] + mu * (u[pos] - u_[pos]);
		v_s[pos].x = v[pos].x + mu * (v[pos].x - v_[pos].x);
		v_s[pos].y = v[pos].y + mu * (v[pos].y - v_[pos].y);
	}
}

void lup::Upsampling::UpdatePrimalVariablesL2(float2* Tp, float* u_, float2* v_, float2* p, float4* q,
	float* a, float* b, float* c,
	float tau, float* eta_u, float* eta_v1, float* eta_v2,
	float alpha0, float alpha1, float* w, float* dw, float mu,
	float* u, float2* v, float* u_s, float2* v_s) {
	dim3 threads(BlockWidth, BlockHeight);
	dim3 blocks(iDivUp(width, threads.x), iDivUp(height, threads.y));
	UpdatePrimalVariablesL2Kernel << < blocks, threads >> > (Tp, u_, v_, p, q,
		a, b, c,
		tau, eta_u, eta_v1, eta_v2,
		alpha0, alpha1, w, dw, mu,
		u, v, u_s, v_s,
		width, height, stride);
}

//void lup::Upsampling::UpsamplingTensorTVGL2(int w, int h, int s, float* u_init, float* depth, float* weight,
//	float* gray, float beta, float gamma, float tgv_alpha, float lambda, int maxits) 
//{
//
//}



