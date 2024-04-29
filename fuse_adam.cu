#pragma once
#include <vector>
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
__global__ void op_adam_single_tensor_kernel(T* t, T* dt, T* mt, T* vt,
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
        t[i] = t[i] - lr * mt_hat / (sqrtf(vt_hat) + epsilon);
    }
}


void
FuseAdamStepCUDASingleTensor(
	torch::Tensor& pp,
	torch::Tensor& grad,
	torch::Tensor& m,
	torch::Tensor& v,
	int t,
    float lr,
    float beta_1,
    float beta_2,
    float epsilon,
    float weight_decay) {
    const auto total_elements = pp.numel(); 

    int num_threads = ONE_DIM_BLOCK_SIZE;
    int num_blocks = (total_elements + ONE_DIM_BLOCK_SIZE - 1) / ONE_DIM_BLOCK_SIZE;


    // NCU Test
    // cudaProfilerStart();
    op_adam_single_tensor_kernel<<<num_blocks, num_threads>>>(
        pp.contiguous().data<float>(),
        grad.contiguous().data<float>(),
        m.contiguous().data<float>(), 
        v.contiguous().data<float>(), 
        lr, beta_1, beta_2, epsilon, weight_decay, t, total_elements);
    // cudaProfilerStop();
    
    cudaError_t error = cudaDeviceSynchronize();
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
    }

	return;
}

__global__ void op_adam_multi_tensor_kernel(TensorInfo* tis, int step, int num_params, int tot_num_elems)
{

    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride_x = blockDim.x * gridDim.x;
    
    for (int global_idx = thread_idx; global_idx < tot_num_elems; global_idx += stride_x) {
        int tensor_idx = num_params - 1;

        #pragma unroll
        for (int j = 0; j < num_params; j++)
        {
            /* code */
            if (global_idx < tis[j].start_idx) { // iterate until g_idx < start_idx
                tensor_idx = j - 1;
                break;
            }
        }
        // if (tensor_idx >= num_params) break;

        TensorInfo* ti = &tis[tensor_idx];
        int local_idx = global_idx - ti->start_idx;
        // if (local_idx >= ti->size) break;

        // if (ti->weight_decay != 0.0) ti->grad_addr[local_idx] += ti->weight_decay * ti->param_addr[local_idx];

        ti->m_addr[local_idx] = ti->beta_1 * ti->m_addr[local_idx] + (1 - ti->beta_1) * ti->grad_addr[local_idx];
        ti->v_addr[local_idx] = ti->beta_2 * ti->v_addr[local_idx] + (1 - ti->beta_2) * ti->grad_addr[local_idx] * ti->grad_addr[local_idx];
        float mt_hat = ti->m_addr[local_idx] / (1 - pow(ti->beta_1, step));
        float vt_hat = ti->v_addr[local_idx] / (1 - pow(ti->beta_2, step));
        ti->param_addr[local_idx] -= ti->lr * mt_hat / (sqrtf(vt_hat) + ti->epsilon);
    }
}



// std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
void
FuseAdamStepCUDAMultiTensor(
	std::vector<torch::Tensor> pp,
	std::vector<torch::Tensor> grad,
	std::vector<torch::Tensor> m,
	std::vector<torch::Tensor> v,
	int step,
    std::vector<float> lr,
    std::vector<float> beta_1,
    std::vector<float> beta_2,
    std::vector<float> epsilon,
    std::vector<float> weight_decay) {

    int num_params = pp.size();
    int tot_num_elems = 0;
    TensorInfo* tis = new TensorInfo[num_params];

    for (int i = 0; i < num_params; i++) {
        tis[i].param_addr = pp[i].data<float>();
        tis[i].grad_addr = grad[i].data<float>();
        tis[i].m_addr = m[i].data<float>();
        tis[i].v_addr = v[i].data<float>();
        tis[i].lr = lr[i];
        tis[i].beta_1 = beta_1[i];
        tis[i].beta_2 = beta_2[i];
        tis[i].epsilon = epsilon[i];
        tis[i].weight_decay = weight_decay[i];
        tis[i].start_idx = tot_num_elems;
        tis[i].size = pp[i].numel();
        tot_num_elems += tis[i].size;
    }
    // std::cout << "tot_num_elems: " << tot_num_elems << std::endl;
    // std::cout << "num_params: " << num_params << std::endl;
    TensorInfo* tis_dev = nullptr;
    cudaError_t status;

    // Allocate GPU memory
    status = cudaMalloc((void**)&tis_dev, num_params * sizeof(TensorInfo));
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA malloc failed: %s\n", cudaGetErrorString(status));
    }
    // Copy data from host vector to device
    status = cudaMemcpy(tis_dev, tis, num_params * sizeof(TensorInfo), cudaMemcpyHostToDevice);
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA memcpy failed: %s\n", cudaGetErrorString(status));
    }

    int num_threads = ONE_DIM_BLOCK_SIZE;
    int num_blocks = (tot_num_elems + ONE_DIM_BLOCK_SIZE - 1) / num_threads;

    op_adam_multi_tensor_kernel<<<num_blocks, num_threads>>>(tis_dev, step, num_params, tot_num_elems);


    cudaError_t error = cudaDeviceSynchronize();
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
    }
    cudaFree(tis_dev);
    delete[] tis;
    tis = nullptr;
	return;
}