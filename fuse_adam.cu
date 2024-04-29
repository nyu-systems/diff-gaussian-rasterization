#pragma once
#include "fuse_adam.h"
// #include <cuda_profiler_api.h>

// gt = dt + lambda * t
// mt = beta_1 * mt + (1 - beta_1) * gt
// vt = beta_2 * vt + (1 - beta_2) * gt ^ 2
// mt_hat = mt / (1 - beta_1 ^ step)
// vt_hat = vt / (1 - beta_2 ^ step)
// if amsgrad: not implement now
// out = t - lr * mt_hat / (sqrt(vt_hat) + epsilon)

template <typename T>
__global__ void op_adam_kernel(T* t, T* dt, T* mt, T* vt, T* out,
            float lr, float beta_1, float beta_2, float epsilon, float lambda, int step, int total_elements)
{

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride_x = blockDim.x * gridDim.x;
    for (int i = idx; i < total_elements; i += stride_x) {
        if (lambda != 0.0) dt[i] += lambda * t[i];

        mt[i] = beta_1 * mt[i] + (1 - beta_1) * dt[i];
        vt[i] = beta_2 * vt[i] + (1 - beta_2) * dt[i] * dt[i];
        float mt_hat = mt[i] / (1 - powf(beta_1, step));
        // mt[i] /= (1 - pow(beta_1, step));
        float vt_hat = vt[i] / (1 - powf(beta_2, step));
        // vt[i] /= (1 - pow(beta_2, step));
        out[i] = t[i] - lr * mt_hat / (sqrtf(vt_hat) + epsilon);
    }
}



std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
FuseAdamStepCUDA(
	torch::Tensor& pp,
	torch::Tensor& grad,
	torch::Tensor& m,
	torch::Tensor& v,
	int t,
    float lr,
    float beta_1,
    float beta_2,
    float epsilon,
    float weight_decay) 
{
    const auto total_elements = pp.numel(); 

    int num_threads = ONE_DIM_BLOCK_SIZE;
    int num_blocks = (total_elements + ONE_DIM_BLOCK_SIZE - 1) / ONE_DIM_BLOCK_SIZE;

    torch::Tensor out = torch::zeros_like(pp);

    // NCU Test
    // cudaProfilerStart();
    op_adam_kernel<<<num_blocks, num_threads>>>(
        pp.contiguous().data<float>(),
        grad.contiguous().data<float>(),
        m.contiguous().data<float>(), 
        v.contiguous().data<float>(), 
        out.contiguous().data<float>(), 
        lr, beta_1, beta_2, epsilon, weight_decay, t, total_elements);
    // cudaProfilerStop();
    
    cudaError_t error = cudaDeviceSynchronize();
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
    }

	return std::make_tuple(out, m, v);
}





// template<typename T>
// class FusedAdam {
//     std::vector<torch::Tensor> params;
//     std::vector<torch::Tensor> mt;
//     std::vector<torch::Tensor> vt;

//     float lr;
//     float beta_1;
//     float beta_2;
//     float epsilon;
//     float lambda; // weight_decay
//     // float vt_max = 0.0;
//     float t = 0;

//     public:

//     FusedAdam(std::vector<torch::Tensor> params_, float lr_, float beta_1_, float beta_2_, float epsilon_, float lambda): 
//     params(params_), lr(lr_), beta_1(beta_1_), beta_2(beta_2_), epsilon(epsilon_), lambda(lambda)
//     {
//         std::cout << "hyperparameters: " << std::endl;
//         std::cout << "\tlr: " << lr << ", beta_1: " << beta_1 << ", beta_2: " 
//                    << beta_2 << ", eps: " << epsilon << ", lambda: " << lambda << std::endl;

//     }
    
//     void step() {
//         if (t == 0) {
//             for (const auto& param : params) {
//             mt.push_back(torch::zeros_like(param, torch::device(torch::kCUDA)));
//             vt.push_back(torch::zeros_like(param, torch::device(torch::kCUDA)));
//         }
//         }
//         t += 1;
//         // std::cout << "iter: " << t << std::endl;


//         for (int i = 0; i < params.size(); i++) {
//             torch::Tensor& pp = params[i];
//             torch::Tensor grad = pp.grad();
//             torch::Tensor& m = mt[i];
//             torch::Tensor& v = vt[i];
//             const auto total_elements = pp.numel();

//             int num_threads = ONE_DIM_BLOCK_SIZE;
//             int num_blocks = (total_elements + ONE_DIM_BLOCK_SIZE - 1) / ONE_DIM_BLOCK_SIZE;
//             op_adam_kernel<<<num_blocks, num_threads>>>(
//                 pp.contiguous().data<float>(),
//                 grad.contiguous().data<float>(),
//                 pp.contiguous().data<float>(),
//                 m.contiguous().data<float>(), 
//                 v.contiguous().data<float>(), 
//                 lr, beta_1, beta_2, epsilon, lambda, t, total_elements);
//         }
//         // std::cout << mt[0]->str() << std::endl;
//         // std::cout << vt[0]->str() << std::endl;

//     } 
// };
