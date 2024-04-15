#pragma once

#include "utils/tensor.cuh"

#define REDUCE_BLOCK_SIZE 1024

template <typename T>
class MaxAccumFunc
{
public:
    // This function compares input x with the current accumulated maximum value stored in accum
    // If x is bigger than accum, stores x in accum and stores x's index (ind_x) to ind_accum
    __host__ __device__ void operator()(const T &x, const int &ind_x, T &accum, int &ind_accum)
    {
        // Lab-1: add your code here
        if (x > accum)
        {
            accum = x;
            ind_accum = ind_x;
        }
    }
};

template <typename T>
class SumAccumFunc
{
public:
    // This function adds input x to the current accumulated sum value stored in accum
    // The accumu's value is updated (to add x).  The ind_x and ind_accum arguments are not used.
    __host__ __device__ void operator()(const T &x, const int &ind_x, T &accum, int &ind_accum)
    {
        // Lab-1: add your code here
        accum = accum + x;
        // The ind_x and ind_accum arguments are not used.
    }
};

// This kernel function performs row-wise reduction of the "in" tensor and stores the result in "out" tensor.
// If "get_index" is true, then the index accumulator values are stored in "out_index" and "out" is not touched.
template <typename OpFunc, typename T>
__global__ void op_reduction_kernel_rowwise(OpFunc f, Tensor<T> in, Tensor<T> out, Tensor<int> out_index, bool get_index)
{
    // Lab-1: add your code here

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int c = col; c < in.w; c += stride) {
        if (IndexOutofBound(in, 0, c)) return;
        T accum = Index(in, 0, c);
        int ind_accum = 0;
        for (int r = 1; r < in.h; r++)
        {
            f(Index(in, r, c), r, accum, ind_accum);
        }
        if (get_index)
        {
            Index(out_index, 0, c) = ind_accum;
        }
        else
        {
            Index(out, 0, c) = accum;
        }
    }
}

// This kernel function performs column-wise reduction of the "in" tensor and stores the result in "out" tensor.
// If "get_index" is true, then the index accumulator values are stored in "out_index" and "out" is not touched.
template <typename OpFunc, typename T>
__global__ void op_reduction_kernel_colwise(OpFunc f, Tensor<T> in, Tensor<T> out, Tensor<int> out_index, bool get_index)
{
    // Lab-1: add your code here
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int r = row; r < in.h; r += stride) {
        if (IndexOutofBound(in, r, 0)) return;
        T accum = Index(in, r, 0);
        int ind_accum = 0;
        for (int c = 1; c < in.w; c++)
        {
            f(Index(in, r, c), c, accum, ind_accum);
        }
        if (get_index)
        {
            Index(out_index, r, 0) = ind_accum;
        }
        else
        {
            Index(out, r, 0) = accum;
        }
    }
}

template <typename OpFunc, typename T>
void op_reduction_gpu(OpFunc f, const Tensor<T> &in, Tensor<T> &out, Tensor<int> &out_index, bool get_index = false)
{
    int out_h = out.h;
    if (!get_index)
    {
        assert((out.h == 1 && in.w == out.w) || (out.w == 1 && in.h == out.h));
    }
    else
    {
        out_h = out_index.h;
        assert((out_index.h == 1 && in.w == out_index.w) || (out_index.w == 1 && in.h == out_index.h));
    }

    Tensor<T> in_d = in.toDevice();
    Tensor<T> out_d = out.toDevice();
    Tensor<int> out_idx_d = out_index.toDevice();

    if (in.h > out_h)
    {
        // Lab-1: add your code here to launch op_reduction_kernel_colwise
        // delete assert(0) when you are finished
        // dim3 dimBlock(REDUCE_BLOCK_SIZE, REDUCE_BLOCK_SIZE);
        // dim3 dimGrid((in.h + REDUCE_BLOCK_SIZE - 1) / REDUCE_BLOCK_SIZE, (in.w + REDUCE_BLOCK_SIZE - 1) / REDUCE_BLOCK_SIZE);
        // int numGroups = (in.w + REDUCE_BLOCK_SIZE - 1) / REDUCE_BLOCK_SIZE;
        int dimBlock = min(REDUCE_BLOCK_SIZE, in.w);
        op_reduction_kernel_rowwise<<<1, dimBlock>>>(f, in_d, out_d, out_idx_d, get_index);
    }
    else
    {
        // Lab-1: add your code here to launch op_reduction_kernel_rowwise
        // delete assert(0) when you are finished
        // dim3 dimBlock(REDUCE_BLOCK_SIZE, REDUCE_BLOCK_SIZE);
        // dim3 dimGrid((in.h + REDUCE_BLOCK_SIZE - 1) / REDUCE_BLOCK_SIZE, (in.w + REDUCE_BLOCK_SIZE - 1) / REDUCE_BLOCK_SIZE);
        // int numGRoups = (in.h + REDUCE_BLOCK_SIZE - 1) / REDUCE_BLOCK_SIZE;
        int dimBlock = min(REDUCE_BLOCK_SIZE, in.h);
        op_reduction_kernel_colwise<<<1, dimBlock>>>(f, in_d, out_d, out_idx_d, get_index);
    }
}

template <typename T>
void op_sum(const Tensor<T> &in, Tensor<T> &out)
{
    Tensor<int> out_index;
    SumAccumFunc<T> f;
    if (in.on_device && out.on_device)
    {
        op_reduction_gpu(f, in, out, out_index, false);
    }
    else
        assert(0);
}

template <typename T>
void op_argmax(const Tensor<T> &in, Tensor<int> &out_index)
{
    Tensor<T> out;
    MaxAccumFunc<T> f;
    if (in.on_device && out_index.on_device)
    {
        op_reduction_gpu(f, in, out, out_index, true);
    }
    else
        assert(0);
}
