#pragma once
#include <torch/extension.h>

#define ONE_DIM_BLOCK_SIZE 256


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
	float weight_decay);