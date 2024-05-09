#pragma once
#include <torch/extension.h>

#define ADAM_BLOCK_SIZE 512

#define MAX_NUM_BLOCK 320
#define MAX_NUM_PARAMS_PER_CHUNK 64
#define ILP 4

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
	float weight_decay);

// struct TensorInfo {
//     float* param_addr;
//     float* grad_addr;
//     float* m_addr;
//     float* v_addr;
//     int size;
//     int start_idx;
//     float lr;
//     float beta_1;
//     float beta_2;
//     float epsilon;
//     float weight_decay;
// };

struct TensorInfo {                                               // Total ~17*4*64
    float* param_addr[MAX_NUM_PARAMS_PER_CHUNK];
    float* grad_addr[MAX_NUM_PARAMS_PER_CHUNK];
    float* m_addr[MAX_NUM_PARAMS_PER_CHUNK];
    float* v_addr[MAX_NUM_PARAMS_PER_CHUNK];
    long size[MAX_NUM_PARAMS_PER_CHUNK];
    long start_idx[MAX_NUM_PARAMS_PER_CHUNK];
    float lr[MAX_NUM_PARAMS_PER_CHUNK];
    float beta_1[MAX_NUM_PARAMS_PER_CHUNK];
    float beta_2[MAX_NUM_PARAMS_PER_CHUNK];
    float epsilon[MAX_NUM_PARAMS_PER_CHUNK];
    float weight_decay[MAX_NUM_PARAMS_PER_CHUNK];
};


// struct TensorMeta {                                                 // Total ~ 1440
//     void* addrs[TENSOR_LIST_DEPTH][MAX_TENSOR_PER_LAUNCH];          // 8 * D * T ~ 1152
//     int size[MAX_TENSOR_PER_LAUNCH];                                // 4 * T ~ 144
//     unsigned char tensor_to_group[MAX_TENSOR_PER_LAUNCH];           // 2 * T ~ 64
//     float lr[MAX_GROUP_PER_LAUNCH];                                 // 5 * 4 * G ~ 80
//     float beta_1[MAX_GROUP_PER_LAUNCH];
//     float beta_2[MAX_GROUP_PER_LAUNCH];
//     float epsilon[MAX_GROUP_PER_LAUNCH];
//     float weight_decay[MAX_GROUP_PER_LAUNCH];
//     int block_to_chunk[MAX_NUM_BLOCKS];                             // 4 * 320 ~ 1280
// };

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
    int ADAM_CHUNK_SIZE);