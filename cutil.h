/* Copyright © 2013 Paulo Roberto Urio <paulourio@gmail.com> */

#ifndef CUTIL_H_
#define CUTIL_H_

#include <cstdio>
#include <cstdlib>

#include <curand.h>

#define cudaSafeCall(func) _cudaSafeCall(func, __FILE__, __LINE__)

extern inline void _cudaSafeCall(const cudaError_t ret, const char *file,
        const int line)
{
    if (ret != cudaSuccess) {
        fprintf(stderr,
                "cudaSafeCall() failed at %s:%i: %s (err=%d)\n",
                file,
                line,
                cudaGetErrorString(ret),
                ret);
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }
}

#define cudaCheckError() _cudaCheckError(__FILE__, __LINE__)

extern inline void _cudaCheckError(const char *file, const int line)
{
#ifndef NDEBUG
    cudaDeviceSynchronize();
#endif
    if (cudaPeekAtLastError() != cudaSuccess) {
        fprintf(stderr,
                "cudaCheckError() failed at %s:%i: %s\n",
                file,
                line,
                cudaGetErrorString(cudaGetLastError()));
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }
}

#define curandSafeCall(func) _curandSafeCall(func, __FILE__, __LINE__)

extern inline void _curandSafeCall(const curandStatus_t ret, const char *file,
        const int line)
{
    if (ret != CURAND_STATUS_SUCCESS) {
        fprintf(stderr,
                "curandSafeCall() failed at %s:%i: (err=%d)\n",
                file,
                line,
                ret);
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }
}

#endif  // CUTIL_H_
