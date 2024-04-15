#pragma once
#include "utils/tensor.cuh"

// This function calculates the cross_entropy loss from the "logits" tensor for a batch of training innput
// and the batch's corresponding "target" label tensor and returns the average loss of the batch.
// It also returns the gradient of the logits tensor.
template <typename T>
T op_cross_entropy_loss(const Tensor<T> &logits, const Tensor<char> &targets,
                        Tensor<T> &d_logits)
{
    assert(logits.h == targets.h && logits.h == d_logits.h);
    assert(logits.w == d_logits.w);
    assert(targets.w == 1);

    assert(logits.on_device && targets.on_device && d_logits.on_device);

    // Lab-2: please add your code here.
    // You need to define separate GPU kernel function(s) and launch them here
    // In order to calculate d_logits, you should derive what its values should be
    // symbolically.

    Tensor<T> loss{logits.h, 1, true};
    Tensor<T> result{1, 1, true};
    Tensor<int> idx{1, 1, true};
    op_const_init(loss, 0.0);
    op_const_init(result, 0.0);
    op_const_init(idx, 0.0);

    int dimBlock = min(1024, logits.h);
    // dim3 dimGrid((C.h + MM_BLOCK_SIZE - 1) / MM_BLOCK_SIZE, (C.w + MM_BLOCK_SIZE - 1) / MM_BLOCK_SIZE);
    op_ce_kernel<<<1, dimBlock>>>(logits, targets, d_logits, loss);
    SumAccumFunc<T> sumFunc;
    op_reduction_gpu(sumFunc, loss, result, idx, false);

    Tensor<T> result_h = result.toHost();
    // std::cout << "ce_loss " << Index(result_h, 0, 0) << std::endl;
    // std::cout << "ce_loss " << loss.toHost().range() << std::endl;

    // assert(Index(result_h, 0, 0) > 0);
    return Index(result_h, 0, 0);
}

template <typename T>
__global__ void op_ce_kernel(Tensor<T> logits, Tensor<char> targets, Tensor<T> d_logits, Tensor<T> loss)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0; // sum of exps
    float max_logit = Index(logits, row, 0);

    for (int i = 0; i < logits.w; i++)
    { // get max_logits
        if (max_logit < Index(logits, row, i))
        {
            max_logit = Index(logits, row, i);
        }
    }

    for (int i = 0; i < logits.w; i++)
    {
        Index(logits, row, i) = exp(Index(logits, row, i) - max_logit); // safe version of softmax
        sum += Index(logits, row, i);
    }

    for (int i = 0; i < logits.w; i++)
    {
        Index(logits, row, i) /= sum; // softmax
    }

    for (int i = 0; i < logits.w; i++)
    {
        Index(loss, row, 0) -= log(Index(logits, row, i)) * (i == Index(targets, row, 0) ? 1 : 0) / logits.h;
    }
    for (int i = 0; i < logits.w; i++)
    {
        Index(d_logits, row, i) = (Index(logits, row, i) - (i == Index(targets, row, 0) ? 1 : 0)) / logits.h; // softmax
    }
}
