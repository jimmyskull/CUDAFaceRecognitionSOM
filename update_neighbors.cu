#include <opencv2/core/core.hpp>

#include "global.h"

#include "cutil.h"

#define RADIUS 15.0
#define LEARNING_RATE 0.5
#define ALPHA_DECAY 3

// Eq 6
__device__ float neighborhood_radius(float epoch) {
	return RADIUS * expf(-1.0 * (epoch / ITERATION_NUMBER) * log(RADIUS + 0.0));
}

// Eq 5
__device__ float neighborhood_kernel(float epoch, float2 best, float2 current) {
	float x = current.x - best.x;
	float y = current.y - best.y;
	float radius = neighborhood_radius(epoch);
	float total = (x * x + y * y) / (2.0 * radius * radius);
	return expf(-1.0 * total);
}

__device__ float learning_rate() {
	return LEARNING_RATE * expf(-1.0 * ALPHA_DECAY * (1.0 / ITERATION_NUMBER));
}

__global__ void cuda_update_neighborhood(float* som, const uchar* input,
		float epoch, float2 bestpos) {
	unsigned int x = blockIdx.x;
	unsigned int y = blockIdx.y;
	float2 neuronpos;
	neuronpos.x = (float) x;
	neuronpos.y = (float) y;

	float kernel = neighborhood_kernel(epoch, bestpos, neuronpos);
	float learning = learning_rate();

	unsigned int blk = blockIdx.y * N + blockIdx.x;
	unsigned int tid = threadIdx.x;

	unsigned int neuron = blk * (blockDim.x * blockDim.y)
			+ threadIdx.x * blockDim.x;
	float* weights = &som[neuron];

	for (unsigned int i = 0; i < DIM; ++i) {
		float w = weights[i];
		float step = learning * kernel * (((float) input[tid * N + i]) - w);
		weights[i] = w + step;
	}
}

__host__ void update_neighborhood(const uchar* input, float epoch,
		cv::Point2f best) {
	dim3 grid(N, N, 1);
	dim3 block(DIM, 1, 1);
	float2 best_pos;
	best_pos.x = (float) best.x;
	best_pos.y = (float) best.y;

	cuda_update_neighborhood<<<grid, block>>>(devSOM, input, epoch, best_pos);
	cudaCheckError();
}

