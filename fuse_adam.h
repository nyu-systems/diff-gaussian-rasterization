#pragma once
#include <torch/extension.h>

#define ADAM_BLOCK_SIZE 512

#define MAX_NUM_BLOCK 320
#define MAX_NUM_PARAMS_PER_CHUNK 60 // 60 is max to ensure mem of args < 4KB
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

struct TensorInfo {                                               // Total: (6*8+5*4) * T ~ 68 * 60 = 4KB - 16
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