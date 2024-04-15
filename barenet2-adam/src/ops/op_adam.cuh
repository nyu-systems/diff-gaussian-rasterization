#pragma once
#include "utils/tensor.cuh"

template <typename T>
void op_adam(const Tensor<T> &t, const Tensor<T> &dt, Tensor<T> &out, float lr, float beta_1, float beta_2, 
            float epsilon, float lambda, Tensor<T> &mt, Tensor<T> &vt, int step)
{

    assert(out.h == t.h && out.w == t.w);
    assert(t.h == dt.h && t.w == dt.w);


    // int dimBlock = min(1024, logits.h);
    // dim3 dimGrid((C.h + MM_BLOCK_SIZE - 1) / MM_BLOCK_SIZE, (C.w + MM_BLOCK_SIZE - 1) / MM_BLOCK_SIZE);
    op_adam_kernel<<<32, 32>>>(t, dt, out, lr, beta_1, beta_2, 
            epsilon, lambda, mt, vt, step);


}

// gt = dt + lambda * t
// mt = beta_1 * mt + (1 - beta_1) * gt
// vt = beta_2 * vt + (1 - beta_2) * gt ^ 2
// mt_hat = mt / (1 - beta_1 ^ step)
// vt_hat = vt / (1 - beta_2 ^ step)
// if amsgrad: not implement now
// out = t - lr * mt_hat / (sqrt(vt_hat) + epsilon)

template <typename T>
__global__ void op_adam_kernel(Tensor<T> t, Tensor<T> dt, Tensor<T> out, float lr, float beta_1, float beta_2, 
            float epsilon, float lambda, Tensor<T> mt, Tensor<T> vt, int step)
{

    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    int stride_x = blockDim.x * gridDim.x;
    int stride_y = blockDim.y * gridDim.y;

    for (int i = row; i < t.h; i += stride_x) {
        for (int j = col; j < t.w; j += stride_y) {
            Index(dt, i, j) += lambda * Index(t, i, j);
            Index(mt, i, j) = beta_1 * Index(mt, i, j) + (1 - beta_1) * Index(dt, i, j);
            Index(vt, i, j) = beta_2 * Index(vt, i, j) + (1 - beta_2) * Index(dt, i, j) * Index(dt, i, j);
            float mt_hat = Index(mt, i, j) / (1 - pow(beta_1, step));
            // Index(mt, i, j) /= (1 - pow(beta_1, step));
            float vt_hat = Index(vt, i, j) / (1 - pow(beta_2, step));
            // Index(vt, i, j) /= (1 - pow(beta_2, step));
            Index(out, i, j) = Index(t, i, j) - lr * mt_hat / (sqrt(vt_hat) + epsilon);
        }
    }
}
