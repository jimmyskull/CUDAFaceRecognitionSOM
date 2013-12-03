#include <opencv2/core/core.hpp>

#include <thrust/device_vector.h>
#include <thrust/extrema.h>

#include "global.h"

#include "cutil.h"

struct NeuronActivation {
	int x, y;
	float distance;
};

struct LowestDistanceNeuron: public thrust::binary_function<NeuronActivation,
		NeuronActivation, bool> {
	__device__ bool operator()(const NeuronActivation& a,
			const NeuronActivation& b) const {
		return (a.distance < b.distance);
	}
};

thrust::device_vector<NeuronActivation> devActivations(N * N);

#define Vj 1

__global__ void cuda_best_neuron(const float* som, const uchar* input,
		NeuronActivation* results) {
	extern __shared__ float s_distance[];

	unsigned int blk = blockIdx.y * N + blockIdx.x;
	unsigned int tid = threadIdx.x;

	s_distance[tid] = 0.0;
	__syncthreads();

	if (tid < DIM) {
		unsigned int neuron = blk * (blockDim.x * blockDim.y)
				+ threadIdx.x * blockDim.x;
		const float* weights = &som[neuron];
		float distance = 0.0;
		for (unsigned int i = 0; i < DIM; ++i) {
			distance += Vj * fabs(*weights++ - ((float) input[tid * N + i]));
		}
		s_distance[tid] = distance;
	}
	__syncthreads();

	for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
		if (tid < s && s_distance[tid] > s_distance[tid + s])
			s_distance[tid] += s_distance[tid + s];
		__syncthreads();
	}

	if (tid < 16) {
		s_distance[tid] += s_distance[tid + 16];
		s_distance[tid] += s_distance[tid + 8];
		s_distance[tid] += s_distance[tid + 4];
		s_distance[tid] += s_distance[tid + 2];
		s_distance[tid] += s_distance[tid + 1];
	}

	if (tid == 0) {
		NeuronActivation& r = results[blk];
		r.x = blockIdx.x;
		r.y = blockIdx.y;
		r.distance = s_distance[0];
	}
}

__host__ cv::Point2f find_best(const uchar* input) {
	dim3 grid(N, N, 1);
	dim3 block(GetUpperBoundPowerOfTwo(DIM), 1, 1);
	size_t shared = DIM * sizeof(float);

	NeuronActivation *devPtr = thrust::raw_pointer_cast(devActivations.data());

	cuda_best_neuron<<<grid, block, shared>>>(devSOM, input, devPtr);
	cudaCheckError();

	NeuronActivation best;
	best = *thrust::min_element(devActivations.begin(),
			devActivations.end(), LowestDistanceNeuron());

	return cv::Point2f(best.x, best.y);
}
