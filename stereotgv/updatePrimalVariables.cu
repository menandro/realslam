#include "stereotgv.h"

__global__ void UpdatePrimalVariablesL2Kernel(float2* Tp, float* u_, float2* v_, float2* p, float4* q,
	float* a, float* b, float* c,
	float tau, float* eta_u, float* eta_v1, float* eta_v2,
	float alpha0, float alpha1, float mu,
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

		// Thresholding
		float uhat = u_[pos] + tau_eta_u * div_p;
		float dun = (uhat - u[pos]);

		//u = (u_ + tau_eta_u.*(alpha1.*div_p + dw)). / (1 + tau_eta_u.*w);
		//u[pos] = (u_[pos] + tau_eta_u * (alpha1 * div_p + dw[pos])) / (1 + tau_eta_u * w[pos]);

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

void StereoTgv::UpdatePrimalVariables(float2* Tp, float* u_, float2* v_, float2* p, float4* q,
	float* a, float* b, float* c,
	float tau, float* eta_u, float* eta_v1, float* eta_v2,
	float alpha0, float alpha1, float mu,
	int w, int h, int s,
	float* u, float2* v, float* u_s, float2* v_s) {
	dim3 threads(BlockWidth, BlockHeight);
	dim3 blocks(iDivUp(w, threads.x), iDivUp(h, threads.y));
	UpdatePrimalVariablesL2Kernel << < blocks, threads >> > (Tp, u_, v_, p, q,
		a, b, c,
		tau, eta_u, eta_v1, eta_v2,
		alpha0, alpha1, mu,
		u, v, u_s, v_s,
		w, h, s);
}