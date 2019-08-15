#include "stereo.h"

__global__
void AddKernel(const float *op1, const float *op2, int count, float *sum)
{
	const int pos = threadIdx.x + blockIdx.x * blockDim.x;

	if (pos >= count) return;

	sum[pos] = op1[pos] + op2[pos];
}

///////////////////////////////////////////////////////////////////////////////
/// \brief add two vectors of size _count_
/// \param[in]  op1   term one
/// \param[in]  op2   term two
/// \param[in]  count vector size
/// \param[out] sum   result
///////////////////////////////////////////////////////////////////////////////

void Stereo::Add(const float *op1, const float *op2, int count, float *sum)
{
	dim3 threads(BlockWidth, BlockHeight);
	//dim3 blocks(iDivUp(w, threads.x), iDivUp(h, threads.y));
	//dim3 threads(256);
	dim3 blocks(iDivUp(count, threads.x));
	AddKernel <<< blocks, threads >>>(op1, op2, count, sum);
}
