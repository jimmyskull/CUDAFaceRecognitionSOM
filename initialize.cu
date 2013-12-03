#include <curand_kernel.h>

#include "global.h"

#include "cutil.h"

__global__ void init_som(unsigned long long int seed, curandState_t* states,
		float* som) {
	unsigned int blk = blockIdx.y * N + blockIdx.x;
	unsigned int tid = threadIdx.x;
	unsigned int glb = blk * DIM + tid;

	curandState_t* state = &states[glb];
	curand_init(seed + glb, 0, 0, state);

	unsigned int neuron = blk * (blockDim.x * blockDim.y)
			+ threadIdx.x * blockDim.x;
	float* weights = &som[neuron];
	for (unsigned int i = 0; i < DIM; ++i) {
		*weights++ = (float) (curand(state) % 255);
	}
}

__host__ void initialize_map() {
	curandState_t* states;

	cudaSafeCall(cudaMalloc(&states, N * N * DIM * sizeof(curandState_t)));
	unsigned long long int seed = time(NULL);

	dim3 grid(N, N, 1);
	dim3 block(DIM, 1, 1);
	init_som<<<grid, block>>>(seed, states, devSOM);
	cudaCheckError();

	cudaSafeCall(cudaFree(states));
}
