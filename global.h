/* Copyright 2013 Paulo Roberto Urio <paulourio@gmail.com> */

#ifndef GLOBAL_H_
#define GLOBAL_H_

// Map size of N x N
#define N 10

// Input image size of DIM x DIM, also the same for weight of each neuron
#define DIM 200

#define ITERATION_NUMBER 2

extern float* devSOM;

static inline unsigned int GetUpperBoundPowerOfTwo(unsigned int n) {
	float p = std::ceil(std::log(n) / std::log(2));
	return static_cast<unsigned int>(std::pow(2, p));
}

#endif  // GLOBAL_H_
