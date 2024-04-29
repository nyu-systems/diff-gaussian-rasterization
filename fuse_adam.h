#pragma once
#include <torch/extension.h>

#define ONE_DIM_BLOCK_SIZE 256


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

struct TensorInfo {
    float* param_addr;
    float* grad_addr;
    float* m_addr;
    float* v_addr;
    int size;
    int start_idx;
    float lr;
    float beta_1;
    float beta_2;
    float epsilon;
    float weight_decay;
};

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
    std::vector<float> weight_decay);