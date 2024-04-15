#pragma once
#include <iostream>
#include <assert.h>
#include <cuda_runtime.h>
#include <curand.h>

#define cudaAssert(err) (cuda_assert((err), __FILE__, __LINE__))
#define curandAssert(err) (curand_assert((err), __FILE__, __LINE__))

void cuda_assert(cudaError_t err, const char *filename, int lineno)
{
    if (err != cudaSuccess)
    {
        std::cerr << filename << ":" << lineno << " CUDA Error " 
            << cudaGetErrorString(err) << std::endl;
        assert(0);
    }
}

void curand_assert(curandStatus_t err, const char *filename, int lineno)
{
    if (err != CURAND_STATUS_SUCCESS)
    {
        std::cerr << filename << ":" << lineno << " CURAND Error "
                  << err << std::endl;
        assert(0);
    }
}