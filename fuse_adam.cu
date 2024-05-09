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
__global__ void op_adam_single_tensor_kernel(T* p, T* g, T* m, T* v,
            float lr, float beta_1, float beta_2, float epsilon, float lambda, int step, int total_elements){

    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride_x = blockDim.x * gridDim.x;
    for (int idx = thread_idx; idx < total_elements; idx += stride_x*ILP) {
      float r_g[ILP];
      float r_p[ILP];
      float r_m[ILP];
      float r_v[ILP];
        #pragma unroll
        for(int ii = 0; ii < ILP; ii++) {
            int i = idx + ii*stride_x;
            if(i < total_elements){
                r_g[ii] = g[i];
                r_p[ii] = p[i];
                r_m[ii] = m[i];
                r_v[ii] = v[i];
            } 
            else {
                r_g[ii] = 0.0;
                r_p[ii] = 0.0;
                r_m[ii] = 0.0;
                r_v[ii] = 0.0;
            }
        }
        #pragma unroll
        for(int ii = 0; ii < ILP; ii++) {
            if (lambda != 0.0) r_g[ii] += lambda * r_p[ii];

            r_m[ii] = beta_1 * r_m[ii] + (1 - beta_1) * r_g[ii];
            r_v[ii] = beta_2 * r_v[ii] + (1 - beta_2) * r_g[ii] * r_g[ii];
            float mt_hat = r_m[ii] / (1 - powf(beta_1, step));
            float vt_hat = r_v[ii] / (1 - powf(beta_2, step));
            r_p[ii] = r_p[ii] - lr * mt_hat / (sqrtf(vt_hat) + epsilon);
        }
        #pragma unroll
        for(int ii = 0; ii < ILP; ii++)
        {
            int i = idx + ii*stride_x;
            if(i < total_elements) {
                p[i] = r_p[ii];
                m[i] = r_m[ii];
                v[i] = r_v[ii];
            }
        }
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

    int num_threads = ADAM_BLOCK_SIZE;
    int num_blocks = min(MAX_NUM_BLOCK, (int) (total_elements + ADAM_BLOCK_SIZE - 1) / ADAM_BLOCK_SIZE);


    // NCU Test
    // cudaProfilerStart();
    op_adam_single_tensor_kernel<<<num_blocks, num_threads>>>(
        pp.contiguous().data<float>(),
        grad.contiguous().data<float>(),
        m.contiguous().data<float>(), 
        v.contiguous().data<float>(), 
        lr, beta_1, beta_2, epsilon, weight_decay, t, total_elements);
    // cudaProfilerStop();
    
    // cudaError_t error = cudaDeviceSynchronize();
    // if (error != cudaSuccess) {
    //     fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
    // }

	return;
}

__global__ void op_adam_multi_tensor_kernel(TensorInfo tis, int step, int num_params, int tot_num_elems)
{

    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride_x = blockDim.x * gridDim.x;
    
    for (int idx = thread_idx; idx < tot_num_elems; idx += stride_x*ILP) {
        int param_idx[ILP]; // tensor idx in tensorInfo list

        #pragma unroll
        for (int ii = 0; ii < ILP; ii++) {
            param_idx[ii] = num_params - 1;
        }

        int ii_idx = 0;                // idx within ILP
        int global_idx = idx;
        int j = 0;
        // #pragma unroll
        while(j < num_params){ // iterate until g_idx < start_idx
            if (global_idx < tis.start_idx[j]) {
                param_idx[ii_idx] = j - 1;
                ii_idx += 1;

                if (ii_idx >= ILP) break;
                else global_idx += stride_x;
            }
            else {
                j += 1;
            }
        }

        float r_p[ILP];
        float r_g[ILP];
        float r_m[ILP];
        float r_v[ILP];
        float r_l[ILP];
        float r_b1[ILP];
        float r_b2[ILP];
        float r_e[ILP];
        float r_w[ILP];

        #pragma unroll
        for(int ii = 0; ii < ILP; ii++) {
            int i = idx + stride_x*ii - tis.start_idx[param_idx[ii]];
            if(i < tis.size[param_idx[ii]]){
                r_p[ii] = tis.param_addr[param_idx[ii]][i];
                r_g[ii] = tis.grad_addr[param_idx[ii]][i];
                r_m[ii] = tis.m_addr[param_idx[ii]][i];
                r_v[ii] = tis.v_addr[param_idx[ii]][i];
                r_l[ii] = tis.lr[param_idx[ii]];
                r_b1[ii] = tis.beta_1[param_idx[ii]];
                r_b2[ii] = tis.beta_2[param_idx[ii]];
                r_e[ii] = tis.epsilon[param_idx[ii]];
                r_w[ii] = tis.weight_decay[param_idx[ii]];
            } 
            else {
                r_g[ii] = 0.0;
                r_p[ii] = 0.0;
                r_m[ii] = 0.0;
                r_v[ii] = 0.0;
                r_l[ii] = 0.0;
                r_b1[ii] = 0.0;
                r_b2[ii] = 0.0;
                r_e[ii] = 0.0;
                r_w[ii] = 0.0;
            }
        }
        #pragma unroll
        for(int ii = 0; ii < ILP; ii++) {
            if (r_w[ii] != 0.0) r_g[ii] += r_w[ii] * r_p[ii];

            r_m[ii] = r_b1[ii] * r_m[ii] + (1 - r_b1[ii]) * r_g[ii];
            r_v[ii] = r_b2[ii] * r_v[ii] + (1 - r_b2[ii]) * r_g[ii] * r_g[ii];
            float mt_hat = r_m[ii] / (1 - powf(r_b1[ii], step));
            float vt_hat = r_v[ii] / (1 - powf(r_b2[ii], step));
            r_p[ii] = r_p[ii] - r_l[ii] * mt_hat / (sqrtf(vt_hat) + r_e[ii]);
        }
        #pragma unroll
        for(int ii = 0; ii < ILP; ii++)
        {
            int i = idx + stride_x*ii - tis.start_idx[param_idx[ii]];
            if(i < tis.size[param_idx[ii]]) {
                tis.param_addr[param_idx[ii]][i]= r_p[ii];
                tis.m_addr[param_idx[ii]][i] = r_m[ii];
                tis.v_addr[param_idx[ii]][i] = r_v[ii];
            }
        }
    }
}

// chuncked version
void
FuseAdamStepCUDAMultiTensor(
	std::vector<std::vector<torch::Tensor>> tensor_list,
	int step,
    std::vector<float> lr,
    std::vector<float> beta_1,
    std::vector<float> beta_2,
    std::vector<float> epsilon,
    std::vector<float> weight_decay,
    std::vector<int> tensor_to_group,
    long tot_num_elems,
    int ADAM_CHUNK_SIZE) {

    std::vector<torch::Tensor> pp = tensor_list[0];
    std::vector<torch::Tensor> grad = tensor_list[1];
    std::vector<torch::Tensor> m = tensor_list[2];
    std::vector<torch::Tensor> v = tensor_list[3];

    int num_params = tensor_list[0].size();
    int tot_num_chunks = (int) (tot_num_elems + ADAM_CHUNK_SIZE - 1) / ADAM_CHUNK_SIZE;

    int num_threads = ADAM_BLOCK_SIZE;
    int num_blocks = min(MAX_NUM_BLOCK, (int) (tot_num_elems + num_threads - 1) / num_threads);

    int param_idx = 0;                  // global idx of params, linear probing
    long param_offset = 0;              // offset in the current parameter
    int chunk_length = 0;               // offset / final length in the current chunk
    int param_idx_in_chunk = 0;         // the idx of params in the current chunk
    TensorInfo tis;

    for (int chunk = 0; chunk < tot_num_chunks; chunk++) {
        for (int t = param_idx; t < min(param_idx + MAX_NUM_PARAMS_PER_CHUNK, num_params); t++) {
            long tensor_length = tensor_list[0][t].numel();

            tis.param_addr[param_idx_in_chunk] = pp[t].data<float>() + param_offset;
            tis.grad_addr[param_idx_in_chunk] = grad[t].data<float>() + param_offset;
            tis.m_addr[param_idx_in_chunk] = m[t].data<float>() + param_offset;
            tis.v_addr[param_idx_in_chunk] = v[t].data<float>() + param_offset;
            tis.start_idx[param_idx_in_chunk] = chunk_length;
            tis.lr[param_idx_in_chunk] = lr[tensor_to_group[t]];
            tis.beta_1[param_idx_in_chunk] = beta_1[tensor_to_group[t]];
            tis.beta_2[param_idx_in_chunk] = beta_2[tensor_to_group[t]];
            tis.epsilon[param_idx_in_chunk] = epsilon[tensor_to_group[t]];
            tis.weight_decay[param_idx_in_chunk] = weight_decay[tensor_to_group[t]];

            if (tensor_length - param_offset >= ADAM_CHUNK_SIZE - chunk_length) {
                tis.size[param_idx_in_chunk] = ADAM_CHUNK_SIZE - chunk_length;
                param_offset += ADAM_CHUNK_SIZE - chunk_length;
                chunk_length = ADAM_CHUNK_SIZE;
                param_idx_in_chunk += 1;
                param_idx = t;
                if (param_offset == tensor_length) {
                    param_offset = 0;
                    param_idx = t + 1;
                }
                break;
            }
            else {
                tis.size[param_idx_in_chunk] = (int) (tensor_length - param_offset);
                chunk_length += (int) (tensor_length - param_offset);
                param_idx_in_chunk += 1;
                param_offset = 0;
                param_idx = t + 1;
                if (param_idx_in_chunk >= MAX_NUM_PARAMS_PER_CHUNK) break;
            }
        }
        // std::cout <<"chunk: " << chunk << ", chunk_length: " << chunk_length << ", param_idx_in_chunk: " << param_idx_in_chunk
        //           << ", param_idx: " << param_idx << ", param_offset: " << param_offset << std::endl;
        // std::cout << tis.size[0] << " : " << tis.size[1] << " : " << tis.size[2]
        //           << " : " << tis.size[3] << " : " << tis.size[4] << " : " << tis.size[5] 
        //           << " : " << tis.size[6] << " : " << tis.size[75] << " : " << tis.size[8] << std::endl;
        // std::cout << tis.start_idx[0] << " : " << tis.start_idx[1] << " : " << tis.start_idx[2]
        //           << " : " << tis.start_idx[3] << " : " << tis.start_idx[4] << " : " << tis.start_idx[5]
        //           << " : " << tis.start_idx[6] << " : " << tis.start_idx[7] << " : " << tis.start_idx[8] << std::endl;

        // int thread_idx = 320*512 - 1;
        // int stride_x = 320*512;
        // for (int idx = thread_idx; idx < chunk_length; idx += stride_x*ILP) {
        //     int r_i[ILP] = {param_idx_in_chunk - 1, param_idx_in_chunk - 1, param_idx_in_chunk - 1, param_idx_in_chunk - 1}; // tensor idx in tensorInfo list
        //     int ii_idx = 0;                // idx within ILP
        //     int global_idx = idx;
        //     int j = 0;
        //     // #pragma unroll
        //     while(j < param_idx_in_chunk){
        //         std::cout << "num_params: " << param_idx_in_chunk << ", global_idx: " << global_idx << ", start_idx: " <<  tis.start_idx[j] << std::endl;
        //         if (global_idx < tis.start_idx[j]) { // iterate until g_idx < start_idx
        //             r_i[ii_idx] = j - 1;
        //             ii_idx += 1;

        //             if (ii_idx >= ILP) break;
        //             else global_idx += stride_x;
        //         }
        //         else {
        //             j += 1;
        //         }
        //     }
        //     for (int ii = 0; ii < ILP; ii++) {
        //         std::cout << ii << " tensor_idx: " << r_i[ii] << "-local_idx: " << idx + stride_x*ii - tis.start_idx[r_i[ii]] << ", start: " << tis.start_idx[r_i[ii]] << ", size: " << tis.size[r_i[ii]] << std::endl;
        //     }
        // }
        op_adam_multi_tensor_kernel<<<num_blocks, num_threads>>>(tis, step, param_idx_in_chunk, chunk_length);
        chunk_length = 0;
        param_idx_in_chunk = 0;
        // cudaError_t error = cudaDeviceSynchronize();
        // if (error != cudaSuccess) {
        //     fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
        // }
    }
}

















    // long num_params = pp.size();
    // int num_groups = group_start_list.size() - 1;
    // TensorInfo* tis = new TensorInfo[num_params];

    // long tot_num_elems = group_start_list[num_groups];
    // long tot_num_chunks = (tot_num_elems + ADAM_CHUNK_SIZE - 1 ) / ADAM_CHUNK_SIZE;
    // // std::cout << tot_num_elems << " : "<< tot_num_chunks  << " : " << ADAM_CHUNK_SIZE << std::endl;

    // int num_threads = ADAM_BLOCK_SIZE;
    // int num_blocks = (tot_num_elems + ADAM_BLOCK_SIZE - 1) / num_threads;
    // TensorInfo* chunk_tis = new TensorInfo[MAX_NUM_PARAMS_PER_CHUNK];
    // TensorInfo* chunk_tis_dev = nullptr;
    // status = cudaMalloc((void**)&chunk_tis_dev, MAX_NUM_PARAMS_PER_CHUNK * sizeof(TensorInfo));

    // long param_start = 0;
    // long param_idx = 0;
    // long param_offset = 0;
    // for (long chunk_idx = 0; chunk_idx < tot_num_chunks; chunk_idx++) {
    //     int num_params_in_chunk = 0;
    //     int chunk_length = 0;

    //     for (int i = param_idx; i < min(num_params, param_idx + MAX_NUM_PARAMS_PER_CHUNK); i++) {
    //         int group_idx = num_groups - 1;
    //         for (int j = group_idx; j < num_groups; j++) {
    //             if (group_start_list[j] > param_start) {
    //                 group_idx = j - 1;
    //                 break;
    //             }
    //         }

    //         chunk_tis[num_params_in_chunk].param_addr = pp[i].data<float>() + (int) param_offset;
    //         chunk_tis[num_params_in_chunk].grad_addr = grad[i].data<float>() + (int) param_offset;
    //         chunk_tis[num_params_in_chunk].m_addr = m[i].data<float>() + (int) param_offset;
    //         chunk_tis[num_params_in_chunk].v_addr = v[i].data<float>() + (int) param_offset;
    //         chunk_tis[num_params_in_chunk].lr = lr[group_idx];
    //         chunk_tis[num_params_in_chunk].beta_1 = beta_1[group_idx];
    //         chunk_tis[num_params_in_chunk].beta_2 = beta_2[group_idx];
    //         chunk_tis[num_params_in_chunk].epsilon = epsilon[group_idx];
    //         chunk_tis[num_params_in_chunk].weight_decay = weight_decay[group_idx];
    //         chunk_tis[num_params_in_chunk].start_idx = chunk_length;


    //         if (ADAM_CHUNK_SIZE - chunk_length <= pp[i].numel() - param_offset) {
    //             chunk_tis[num_params_in_chunk].size = ADAM_CHUNK_SIZE - chunk_length;
    //             param_offset += ADAM_CHUNK_SIZE - chunk_length;
    //             chunk_length = ADAM_CHUNK_SIZE;
    //             break;
    //         }

    //         chunk_length += (int) (pp[i].numel() - param_offset);
    //         param_idx += 1;
    //         param_start += (int) pp[i].numel();
    //         param_offset = 0;
    //         chunk_tis[num_params_in_chunk].size = (int) (pp[i].numel() - param_offset);

    //         num_params_in_chunk += 1;

    //         if (num_params_in_chunk >= MAX_NUM_PARAMS_PER_CHUNK) break;
    //     }
    //     // std::cout << "chunk length: " << chunk_length << std::endl;

    //     status = cudaMemcpy(chunk_tis_dev, chunk_tis, MAX_NUM_PARAMS_PER_CHUNK * sizeof(TensorInfo), cudaMemcpyHostToDevice);

    //     // std::cout << "p start: " << param_start << ", p_idx: "<< param_idx  << ", p_off: " << param_offset << ", num_params: " << num_params_in_chunk << ", chunk length: " << chunk_length << std::endl;


    //     op_adam_multi_tensor_kernel<<<num_blocks, num_threads>>>(chunk_tis_dev, step, num_params_in_chunk, chunk_length);
    //     cudaError_t error = cudaDeviceSynchronize();
    //     if (error != cudaSuccess) {
    //         fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
    //     }
    // }
    // cudaFree(chunk_tis_dev);
    // delete[] chunk_tis;
    // chunk_tis = nullptr;