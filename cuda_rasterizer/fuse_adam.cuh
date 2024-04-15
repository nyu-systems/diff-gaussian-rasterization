#pragma once
#include <torch/extension.h>

// copied from barenet crossentropy
// TODO: replace Tensor<T> with torch::Tensor
//  initialize with torch::full for full presicion
// TODO: Read Pytorch Adam source code, try to write the basic version

void op_cross_entropy_loss(const torch::Tensor &logits, const Tensor<char> &targets,
                        torch::Tensor &d_logits)
{

    op_ce_kernel<<<1, 1024>>>(logits, targets, d_logits, loss);
    return;
}

__global__ void op_ce_kernel(torch::Tensor logits, Tensor<char> targets, torch::Tensor d_logits, torch::Tensor loss)
{

}
