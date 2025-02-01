/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include <math.h>
#include <torch/extension.h>
#include <cstdio>
#include <sstream>
#include <iostream>
#include <tuple>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <memory>
#include "cuda_rasterizer/config.h"
#include "cuda_rasterizer/rasterizer.h"
#include "cuda_rasterizer/auxiliary.h"
#include <fstream>
#include <string>
#include <functional>
#include <cassert>
#include <c10/cuda/CUDAStream.h>

#include <cooperative_groups.h>
namespace cg = cooperative_groups;

std::function<char*(size_t N)> resizeFunctional(torch::Tensor& t) {
    auto lambda = [&t](size_t N) {
        t.resize_({(long long)N});
		return reinterpret_cast<char*>(t.contiguous().data_ptr());
    };
    return lambda;
}

torch::Tensor markVisible(
		torch::Tensor& means3D,
		torch::Tensor& viewmatrix,
		torch::Tensor& projmatrix)
{ 
  const int P = means3D.size(0);
  
  torch::Tensor present = torch::full({P}, false, means3D.options().dtype(at::kBool));
 
  if(P != 0)
  {
	CudaRasterizer::Rasterizer::markVisible(P,
		means3D.contiguous().data<float>(),
		viewmatrix.contiguous().data<float>(),
		projmatrix.contiguous().data<float>(),
		present.contiguous().data<bool>());
  }
  
  return present;
}

torch::Tensor GetSend2GpuCUDA(
    torch::Tensor& means3D,
    torch::Tensor& viewmatrix,
    torch::Tensor& projmatrix)
{
    const int P = means3D.size(0);

    torch::Tensor present = torch::full({P}, false, means3D.options().dtype(at::kBool));
 
    if(P != 0) {
        CudaRasterizer::Rasterizer::getSend2Gpu(
            P,
		    means3D.contiguous().data<float>(),
		    viewmatrix.contiguous().data<float>(),
		    projmatrix.contiguous().data<float>(),
		    present.contiguous().data<bool>());
    }
  
    return present;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
Send2GpuCUDA(
    torch::Tensor& opacities,
    torch::Tensor& scales,
    torch::Tensor& rotations,
    torch::Tensor& features_dc,
    torch::Tensor& features_rest,
    torch::Tensor& mask)
{
    int64_t N = mask.size(0);
    int64_t num_select = mask.to(torch::kInt32).sum().item<int>();
    int64_t opacities_dim1 = opacities.size(1);
    int64_t scales_dim1 = scales.size(1);
    int64_t rotations_dim1 = rotations.size(1);
    int64_t features_dc_dim1 = features_dc.size(1);
    int64_t features_dc_dim2 = features_dc.size(2);
    int64_t features_rest_dim1 = features_rest.size(1);
    int64_t features_rest_dim2 = features_rest.size(2);
    
    torch::Tensor d_opacities = torch::empty({num_select, opacities_dim1}, opacities.options().device(torch::kCUDA));
    torch::Tensor d_scales = torch::empty({num_select, scales_dim1}, scales.options().device(torch::kCUDA));
    torch::Tensor d_rotations = torch::empty({num_select, rotations_dim1}, rotations.options().device(torch::kCUDA));
    torch::Tensor d_features_dc = torch::empty({num_select, features_dc_dim1, features_dc_dim2}, features_dc.options().device(torch::kCUDA));
    torch::Tensor d_features_rest = torch::empty({num_select, features_rest_dim1, features_dc_dim2}, features_rest.options().device(torch::kCUDA));

    // cudaDeviceSynchronize();

    CudaRasterizer::Rasterizer::scattered_transfer(
        'f',
        NULL,
        opacities.contiguous().data<float>(),
        scales.contiguous().data<float>(),
        rotations.contiguous().data<float>(),
        features_dc.contiguous().data<float>(),
        features_rest.contiguous().data<float>(),
        mask.contiguous().data<bool>(),
        0,
        opacities_dim1,
        scales_dim1,
        rotations_dim1,
        features_dc_dim1 * features_dc_dim2,
        features_rest_dim1 * features_rest_dim2,
        N,
        num_select,
        NULL,
        d_opacities.contiguous().data<float>(),
        d_scales.contiguous().data<float>(),
        d_rotations.contiguous().data<float>(),
        d_features_dc.contiguous().data<float>(),
        d_features_rest.contiguous().data<float>(),
        false
    );

    return std::make_tuple(d_opacities, d_scales, d_rotations, d_features_dc, d_features_rest);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
SendCat2GpuCUDA(
    torch::Tensor& parameters,
    torch::Tensor& mask,
    torch::Tensor& dims,
    torch::Tensor& dims_presum_shift,
    torch::Tensor& col2attr)
{
    int64_t N = mask.size(0);
    int64_t num_select = mask.to(torch::kInt32).sum().item<int>();
    int n_col = parameters.size(1);

    auto options = torch::TensorOptions().device(torch::kCUDA);
    torch::Tensor d_opacities = torch::empty({num_select, dims[1].item<int>()}, options);
    torch::Tensor d_scales = torch::empty({num_select, dims[2].item<int>()}, options);
    torch::Tensor d_rotations = torch::empty({num_select, dims[3].item<int>()}, options);
    torch::Tensor d_features_dc = torch::empty({num_select, dims[4].item<int>()}, options);
    torch::Tensor d_features_rest = torch::empty({num_select, dims[5].item<int>()}, options);

    float **h_parameters;
    cudaMallocHost(&h_parameters, 6 * sizeof(float *));
    h_parameters[1] = d_opacities.contiguous().data<float>();
    h_parameters[2] = d_scales.contiguous().data<float>();
    h_parameters[3] = d_rotations.contiguous().data<float>();
    h_parameters[4] = d_features_dc.contiguous().data<float>();
    h_parameters[5] = d_features_rest.contiguous().data<float>();

    float **d_parameters;
    cudaMalloc(&d_parameters, 6 * sizeof(float *));
    cudaMemcpy(d_parameters, h_parameters, 6 * sizeof(float *), cudaMemcpyHostToDevice);

    CudaRasterizer::Rasterizer::cat_transfer(
        'f',
        d_parameters,
        parameters.contiguous().data<float>(),
        mask.contiguous().data<bool>(),
        dims.contiguous().data<int>(),
        dims_presum_shift.contiguous().data<int>(),
        col2attr.contiguous().data<int>(),
        n_col,
        N,
        num_select,
        false
    );

    return std::make_tuple(d_opacities, d_scales, d_rotations, d_features_dc, d_features_rest);
}

template <typename... Args>
void allocateParametersPointers(float**& d_parameters, int n_pointers, Args&&... pointers) {
   cudaMalloc(reinterpret_cast<void**>(&d_parameters), n_pointers * sizeof(float*));
   float* host_pointers[] = {std::forward<Args>(pointers)...};
   cudaMemcpy(d_parameters, host_pointers, n_pointers * sizeof(float*), cudaMemcpyHostToDevice);
}

torch::Tensor SendCat2GpuXYZCUDA(
    torch::Tensor& parameters)
{
	int64_t N = parameters.size(0);

    auto options = torch::TensorOptions().device(torch::kCUDA);
    torch::Tensor d_xyz = torch::empty({N, 3}, options);

    CudaRasterizer::Rasterizer::cat_transfer_xyz(
        d_xyz.contiguous().data<float>(),
        parameters.contiguous().data<float>(),
        N,
        false
    );

    return d_xyz;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
SendCat2GpuOSRCUDA(
    torch::Tensor& parameters,
    torch::Tensor& mask,
    torch::Tensor& mask_indices,
	torch::Tensor& dims,
    torch::Tensor& dims_presum_shift,
    torch::Tensor& col2attr)
{
	int64_t N = mask.size(0);
    int64_t num_select = mask_indices.size(0);

    auto options = torch::TensorOptions().device(torch::kCUDA);
    torch::Tensor d_opacities = torch::empty({num_select, 1}, options);
    torch::Tensor d_scales = torch::empty({num_select, 3}, options);
    torch::Tensor d_rotations = torch::empty({num_select, 4}, options);

	int n_all_col = parameters.size(1);
	int n_load_col = d_opacities.size(1) + d_scales.size(1) + d_rotations.size(1);

    float **d_parameters;
    allocateParametersPointers(d_parameters, 3,
		d_opacities.contiguous().data<float>(),
		d_scales.contiguous().data<float>(),
		d_rotations.contiguous().data<float>());

    CudaRasterizer::Rasterizer::cat_transfer_osr(
        d_parameters,
        parameters.contiguous().data<float>(),
        mask.contiguous().data<bool>(),
		mask_indices.contiguous().data<int64_t>(),
        dims.contiguous().data<int>(),
        dims_presum_shift.contiguous().data<int>(),
        col2attr.contiguous().data<int>(),
        n_all_col,
		n_load_col,
        N,
        num_select,
        false
    );

	cudaDeviceSynchronize();
	cudaFree(d_parameters);

    return std::make_tuple(d_opacities, d_scales, d_rotations);
}

torch::Tensor SendCat2GpuSHSCUDA(
    torch::Tensor& parameters,
    torch::Tensor& mask_indices)
{
	int64_t N = parameters.size(0);
    int64_t num_select = mask_indices.size(0);

    auto options = torch::TensorOptions().device(torch::kCUDA);
    torch::Tensor d_shs = torch::empty({num_select, 48}, options);

	CudaRasterizer::Rasterizer::cat_transfer_shs(
		d_shs.contiguous().data<float>(),
        parameters.contiguous().data<float>(),
		mask_indices.contiguous().data<int64_t>(),
        N,
        num_select,
        false
    );

    return d_shs;
}

torch::Tensor SendSHS2GpuSHSCUDA(
    torch::Tensor& parameters,
    torch::Tensor& mask_indices)
{
    int64_t N = parameters.size(0);
    int64_t num_select = mask_indices.size(0);

    auto options = torch::TensorOptions().device(torch::kCUDA);
    torch::Tensor d_shs = torch::empty({num_select, 48}, options);

    CudaRasterizer::Rasterizer::transfer_shs(
        d_shs.contiguous().data<float>(),
        parameters.contiguous().data<float>(),
        mask_indices.contiguous().data<int64_t>(),
        N,
        num_select,
        false
    );

    return d_shs;
}

__global__ void transfer_shs_cpu2gpu_kernel_stream(
    float *d_shs,
    const float *h_shs,
    const int64_t *rank2id,
    int64_t num_select
) {
    int64_t stride = gridDim.x * blockDim.x;
    int64_t total_elements = num_select * 48;

    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < total_elements; i += stride) {
        int64_t row = i / 48;
        int col = i % 48;

        int64_t offset_srce = rank2id[row] * 48 + col;
        int64_t offset_dest = i;

        d_shs[offset_dest] = h_shs[offset_srce];
    }
}

void SendSHS2GpuStreamCUDA(
    torch::Tensor& d_parameters,
    torch::Tensor& h_parameters,
    torch::Tensor& mask_indices,
    int grid_size,
    int block_size)
{
    int64_t N = h_parameters.size(0);
    int64_t num_select = mask_indices.size(0);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    // const int grid_size = 32;
    // const int grid_size = 16;
    // const int grid_size = 8;
    // const int block_size = 256;
    transfer_shs_cpu2gpu_kernel_stream<<<grid_size, block_size, 0, stream>>>(
        d_parameters.contiguous().data<float>(),
        h_parameters.contiguous().data<float>(),
        mask_indices.contiguous().data<int64_t>(),
        num_select
    );
}

__global__ void transfer_H_shs_cpu2gpu_kernel_stream(
    float *d_shs,
    const float *h_shs,
    const int64_t *host_indices,
    const int64_t *param_indices_from_host,
    int64_t num_select_from_host
) {
    int64_t stride = gridDim.x * blockDim.x;
    int64_t total_elements = num_select_from_host * 48;

    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < total_elements; i += stride) {
        int64_t row = i / 48;
        int col = i % 48;

        int64_t offset_srce = host_indices[row] * 48 + col;
        int64_t offset_dest = param_indices_from_host[row] * 48 + col;

        d_shs[offset_dest] = h_shs[offset_srce];
    }
}

__global__ void transfer_D_shs_cpu2gpu_kernel_stream(
    float *d_shs,
    const float *r_shs,
    const int64_t *host_indices,
    const int64_t *param_indices_from_rtnt,
    int64_t num_select_from_host
) {
    int64_t stride = gridDim.x * blockDim.x;
    int64_t total_elements = num_select_from_host * 48;

    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < total_elements; i += stride) {
        int64_t row = i / 48;
        int col = i % 48;

        int64_t offset_srce = host_indices[row] * 48 + col;
        int64_t offset_dest = param_indices_from_rtnt[row] * 48 + col;

        d_shs[offset_dest] = r_shs[offset_srce];
    }
}

void SendSHS2GpuStreamRetentionCUDA(
    torch::Tensor& d_parameters,
    torch::Tensor& h_parameters,
    torch::Tensor& r_parameters, // on gpu, retent from last iteration
    torch::Tensor& host_indices,
    torch::Tensor& rtnt_indices,
    torch::Tensor& param_indices_from_host,
    torch::Tensor& param_indices_from_rtnt,
    int grid_size_H,
    int block_size_H,
    int grid_size_D,
    int block_size_D)
{
    int64_t N = h_parameters.size(0); // number of all gaussians
    int64_t num_select_from_host = host_indices.size(0);
    int64_t num_select_from_retent = rtnt_indices.size(0);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    transfer_H_shs_cpu2gpu_kernel_stream<<<grid_size_H, block_size_H, 0, stream>>>(
        d_parameters.contiguous().data<float>(),
        h_parameters.contiguous().data<float>(),
        host_indices.contiguous().data<int64_t>(),
        param_indices_from_host.contiguous().data<int64_t>(),
        num_select_from_host
    );

    transfer_D_shs_cpu2gpu_kernel_stream<<<grid_size_D, block_size_D, 0, stream>>>(
        d_parameters.contiguous().data<float>(),
        r_parameters.contiguous().data<float>(),
        rtnt_indices.contiguous().data<int64_t>(),
        param_indices_from_rtnt.contiguous().data<int64_t>(),
        num_select_from_retent
    );
}

__global__ void transfer_H_shs_cpu2gpu_kernel_stream_retention2(
    float *d_shs,
    const float *h_shs,
    const int *filter,
    const int *retention_vec,
    int num_select
) {
    int stride = gridDim.x * blockDim.x;
    int total_elements = num_select * 48;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total_elements; i += stride) {
        int row = filter[i / 48];
        int col = i % 48;

        if (retention_vec[row] == -1) {
            int offset_srce = row * 48 + col;
            int offset_dest = i;

            d_shs[offset_dest] = h_shs[offset_srce];
        }
    }
}

__global__ void transfer_D_shs_cpu2gpu_kernel_stream_retention2(
    float *d_shs,
    const float *r_shs,
    const int *filter,
    const int *retention_vec,
    int num_select
) {
    int stride = gridDim.x * blockDim.x;
    int total_elements = num_select * 48;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total_elements; i += stride) {
        int row = filter[i / 48];
        int col = i % 48;

        if (retention_vec[row] != -1) {
            int offset_srce = retention_vec[row] * 48 + col;
            int offset_dest = i;

            d_shs[offset_dest] = r_shs[offset_srce];
        }
    }
}

void SendSHS2GpuStreamRetention2CUDA(
    torch::Tensor& d_parameters, // shs to fill on device
    torch::Tensor& h_parameters, // shs on host
    torch::Tensor& r_parameters, // on gpu, retent from last iteration
    torch::Tensor& filter, // index filter for this iteration, (d_parameters.shape[0],)
    torch::Tensor& retention_vec, // (n_gaussians,)
    int grid_size_H,
    int block_size_H,
    int grid_size_D,
    int block_size_D)
{
    int N = h_parameters.size(0); // number of all gaussians
    int num_select = filter.size(0); // number of visible gaussians this iter

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    transfer_D_shs_cpu2gpu_kernel_stream_retention2<<<grid_size_D, block_size_D, 0, stream>>>(
        d_parameters.contiguous().data<float>(),
        r_parameters.contiguous().data<float>(),
        filter.contiguous().data<int>(),
        retention_vec.contiguous().data<int>(),
        num_select
    );

    

    transfer_H_shs_cpu2gpu_kernel_stream_retention2<<<grid_size_H, block_size_H, 0, stream>>>(
        d_parameters.contiguous().data<float>(),
        h_parameters.contiguous().data<float>(),
        filter.contiguous().data<int>(),
        retention_vec.contiguous().data<int>(),
        num_select
    );


}

void SendCat2GpuBufferCUDA(
    torch::Tensor& parameters,
    torch::Tensor& mask,
    torch::Tensor& dims,
    torch::Tensor& dims_presum_shift,
    torch::Tensor& col2attr,
    torch::Tensor& d_opacities,
    torch::Tensor& d_scales,
    torch::Tensor& d_rotations,
    torch::Tensor& d_features_dc,
    torch::Tensor& d_features_rest)
{
    int64_t N = mask.size(0);
    int64_t num_select = mask.to(torch::kInt32).sum().item<int>();
    int n_col = parameters.size(1);

    float **h_parameters;
    cudaMallocHost(&h_parameters, 6 * sizeof(float *));
    h_parameters[1] = d_opacities.contiguous().data<float>();
    h_parameters[2] = d_scales.contiguous().data<float>();
    h_parameters[3] = d_rotations.contiguous().data<float>();
    h_parameters[4] = d_features_dc.contiguous().data<float>();
    h_parameters[5] = d_features_rest.contiguous().data<float>();

    float **d_parameters;
    cudaMalloc(&d_parameters, 6 * sizeof(float *));
    cudaMemcpy(d_parameters, h_parameters, 6 * sizeof(float *), cudaMemcpyHostToDevice);

    CudaRasterizer::Rasterizer::cat_transfer(
        'f',
        d_parameters,
        parameters.contiguous().data<float>(),
        mask.contiguous().data<bool>(),
        dims.contiguous().data<int>(),
        dims_presum_shift.contiguous().data<int>(),
        col2attr.contiguous().data<int>(),
        n_col,
        N,
        num_select,
        false
    );
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
Send2CpuCUDA_deprecated(
    torch::Tensor& dmeans3D,
    torch::Tensor& dopacities,
    torch::Tensor& dscales,
    torch::Tensor& drotations,
    torch::Tensor& dfeatures_dc,
    torch::Tensor& dfeatures_rest,
    torch::Tensor& mask)
{
    int64_t N = mask.size(0);
    int64_t num_select = mask.to(torch::kInt32).sum().item<int>();
    int64_t means3D_dim1 = dmeans3D.size(1);
    int64_t opacities_dim1 = dopacities.size(1);
    int64_t scales_dim1 = dscales.size(1);
    int64_t rotations_dim1 = drotations.size(1);
    int64_t features_dc_dim1 = dfeatures_dc.size(1);
    int64_t features_dc_dim2 = dfeatures_dc.size(2);
    int64_t features_rest_dim1 = dfeatures_rest.size(1);
    int64_t features_rest_dim2 = dfeatures_rest.size(2);
    
    torch::Tensor h_dmeans3D = torch::zeros({N, means3D_dim1}, dopacities.options().device(torch::kCPU).pinned_memory(true));
    torch::Tensor h_dopacities = torch::zeros({N, opacities_dim1}, dopacities.options().device(torch::kCPU).pinned_memory(true));
    torch::Tensor h_dscales = torch::zeros({N, scales_dim1}, dscales.options().device(torch::kCPU).pinned_memory(true));
    torch::Tensor h_drotations = torch::zeros({N, rotations_dim1}, drotations.options().device(torch::kCPU).pinned_memory(true));
    torch::Tensor h_dfeatures_dc = torch::zeros({N, features_dc_dim1, features_dc_dim2}, dfeatures_dc.options().device(torch::kCPU).pinned_memory(true));
    torch::Tensor h_dfeatures_rest = torch::zeros({N, features_rest_dim1, features_rest_dim2}, dfeatures_rest.options().device(torch::kCPU).pinned_memory(true));

    assert(h_dmeans3D.is_pinned());
    assert(h_dopacities.is_pinned());
    assert(h_dscales.is_pinned());
    assert(h_drotations.is_pinned());
    assert(h_dfeatures_dc.is_pinned());
    assert(h_dfeatures_rest.is_pinned());

    assert(dmeans3D.device().is_cuda());
    assert(dopacities.device().is_cuda());
    assert(dscales.device().is_cuda());
    assert(drotations.device().is_cuda());
    assert(dfeatures_dc.device().is_cuda());
    assert(dfeatures_rest.device().is_cuda());
    assert(mask.device().is_cuda());
    
    CudaRasterizer::Rasterizer::scattered_transfer(
        'b',
        dmeans3D.contiguous().data<float>(),
        dopacities.contiguous().data<float>(),
        dscales.contiguous().data<float>(),
        drotations.contiguous().data<float>(),
        dfeatures_dc.contiguous().data<float>(),
        dfeatures_rest.contiguous().data<float>(),
        mask.contiguous().data<bool>(),
        means3D_dim1,
        opacities_dim1,
        scales_dim1,
        rotations_dim1,
        features_dc_dim1 * features_dc_dim2,
        features_rest_dim1 * features_rest_dim2,
        N,
        num_select,
        h_dmeans3D.contiguous().data<float>(),
        h_dopacities.contiguous().data<float>(),
        h_dscales.contiguous().data<float>(),
        h_drotations.contiguous().data<float>(),
        h_dfeatures_dc.contiguous().data<float>(),
        h_dfeatures_rest.contiguous().data<float>(),
        false
    );

    cudaDeviceSynchronize();

    return std::make_tuple(h_dmeans3D, h_dopacities, h_dscales, h_drotations, h_dfeatures_dc, h_dfeatures_rest);
}

void Send2CpuCUDA(
    torch::Tensor& dmeans3D,
    torch::Tensor& dopacities,
    torch::Tensor& dscales,
    torch::Tensor& drotations,
    torch::Tensor& dfeatures_dc,
    torch::Tensor& dfeatures_rest,
    torch::Tensor& mask,
    torch::Tensor& h_dmeans3D,
    torch::Tensor& h_dopacities,
    torch::Tensor& h_dscales,
    torch::Tensor& h_drotations,
    torch::Tensor& h_dfeatures_dc,
    torch::Tensor& h_dfeatures_rest)
{
    int64_t N = mask.size(0);
    int64_t num_select = mask.to(torch::kInt32).sum().item<int>();
    int64_t means3D_dim1 = dmeans3D.size(1);
    int64_t opacities_dim1 = dopacities.size(1);
    int64_t scales_dim1 = dscales.size(1);
    int64_t rotations_dim1 = drotations.size(1);
    int64_t features_dc_dim1 = dfeatures_dc.size(1);
    int64_t features_dc_dim2 = dfeatures_dc.size(2);
    int64_t features_rest_dim1 = dfeatures_rest.size(1);
    int64_t features_rest_dim2 = dfeatures_rest.size(2);

    assert(h_dmeans3D.is_pinned());
    assert(h_dopacities.is_pinned());
    assert(h_dscales.is_pinned());
    assert(h_drotations.is_pinned());
    assert(h_dfeatures_dc.is_pinned());
    assert(h_dfeatures_rest.is_pinned());

    assert(dmeans3D.device().is_cuda());
    assert(dopacities.device().is_cuda());
    assert(dscales.device().is_cuda());
    assert(drotations.device().is_cuda());
    assert(dfeatures_dc.device().is_cuda());
    assert(dfeatures_rest.device().is_cuda());
    assert(mask.device().is_cuda());
    
    CudaRasterizer::Rasterizer::scattered_transfer(
        'b',
        dmeans3D.contiguous().data<float>(),
        dopacities.contiguous().data<float>(),
        dscales.contiguous().data<float>(),
        drotations.contiguous().data<float>(),
        dfeatures_dc.contiguous().data<float>(),
        dfeatures_rest.contiguous().data<float>(),
        mask.contiguous().data<bool>(),
        means3D_dim1,
        opacities_dim1,
        scales_dim1,
        rotations_dim1,
        features_dc_dim1 * features_dc_dim2,
        features_rest_dim1 * features_rest_dim2,
        N,
        num_select,
        h_dmeans3D.contiguous().data<float>(),
        h_dopacities.contiguous().data<float>(),
        h_dscales.contiguous().data<float>(),
        h_drotations.contiguous().data<float>(),
        h_dfeatures_dc.contiguous().data<float>(),
        h_dfeatures_rest.contiguous().data<float>(),
        false
    );

    cudaDeviceSynchronize();
}

torch::Tensor Send2CpuCatCUDA(
    torch::Tensor& dmeans3D,
    torch::Tensor& dopacities,
    torch::Tensor& dscales,
    torch::Tensor& drotations,
    torch::Tensor& dfeatures_dc,
    torch::Tensor& dfeatures_rest,
    torch::Tensor& mask,
    torch::Tensor& dims,
    torch::Tensor& dims_presum_shift,
    torch::Tensor& col2attr)
{
    int64_t N = mask.size(0);
    int64_t num_select = mask.to(torch::kInt32).sum().item<int>();
    int n_col = dims.sum().item<int>();

    torch::Tensor dparameters = torch::empty({N, dims.size(0)});

    float **h_param_ptrs;
    cudaMallocHost(&h_param_ptrs, 6 * sizeof(float *));
    h_param_ptrs[0] = dmeans3D.contiguous().data<float>();
    h_param_ptrs[1] = dopacities.contiguous().data<float>();
    h_param_ptrs[2] = dscales.contiguous().data<float>();
    h_param_ptrs[3] = drotations.contiguous().data<float>();
    h_param_ptrs[4] = dfeatures_dc.contiguous().data<float>();
    h_param_ptrs[5] = dfeatures_rest.contiguous().data<float>();

    float **d_param_ptrs;
    cudaMalloc(&d_param_ptrs, 6 * sizeof(float *));
    cudaMemcpy(d_param_ptrs, h_param_ptrs, 6 * sizeof(float *), cudaMemcpyHostToDevice);

    CudaRasterizer::Rasterizer::cat_transfer(
        'b',
        d_param_ptrs,
        dparameters.contiguous().data<float>(),
        mask.contiguous().data<bool>(),
        dims.contiguous().data<int>(),
        dims_presum_shift.contiguous().data<int>(),
        col2attr.contiguous().data<int>(),
        n_col,
        N,
        num_select,
        false
    );

    return dparameters;
}

void Send2CpuCatBufferCUDA(
    torch::Tensor& dmeans3D,
    torch::Tensor& dopacities,
    torch::Tensor& dscales,
    torch::Tensor& drotations,
    torch::Tensor& dfeatures_dc,
    torch::Tensor& dfeatures_rest,
    torch::Tensor& mask,
    torch::Tensor& dims,
    torch::Tensor& dims_presum_shift,
    torch::Tensor& col2attr,
    torch::Tensor& h_dparameters)
{
    int64_t N = mask.size(0);
    int64_t num_select = mask.to(torch::kInt32).sum().item<int>();
    int n_col = dims.sum().item<int>();

    float **h_param_ptrs;
    cudaMallocHost(&h_param_ptrs, 6 * sizeof(float *));
    h_param_ptrs[0] = dmeans3D.contiguous().data<float>();
    h_param_ptrs[1] = dopacities.contiguous().data<float>();
    h_param_ptrs[2] = dscales.contiguous().data<float>();
    h_param_ptrs[3] = drotations.contiguous().data<float>();
    h_param_ptrs[4] = dfeatures_dc.contiguous().data<float>();
    h_param_ptrs[5] = dfeatures_rest.contiguous().data<float>();

    float **d_param_ptrs;
    cudaMalloc(&d_param_ptrs, 6 * sizeof(float *));
    cudaMemcpy(d_param_ptrs, h_param_ptrs, 6 * sizeof(float *), cudaMemcpyHostToDevice);

    CudaRasterizer::Rasterizer::cat_transfer(
        'b',
        d_param_ptrs,
        h_dparameters.contiguous().data<float>(),
        mask.contiguous().data<bool>(),
        dims.contiguous().data<int>(),
        dims_presum_shift.contiguous().data<int>(),
        col2attr.contiguous().data<int>(),
        n_col,
        N,
        num_select,
        false
    );
}

void Send2CpuCatBufferOSRSHSCUDA(
    torch::Tensor& dmeans3D,
    torch::Tensor& dopacities,
    torch::Tensor& dscales,
    torch::Tensor& drotations,
    torch::Tensor& dshs,
    torch::Tensor& infrustum_radii_opacities_filter_indices,
    torch::Tensor& send2gpu_final_filter_indices,
	torch::Tensor& dims,
    torch::Tensor& dims_presum_shift,
    torch::Tensor& col2attr,
    torch::Tensor& h_dparameters,
    bool accum)
{
	int64_t N = h_dparameters.size(0);
	int64_t num_select = send2gpu_final_filter_indices.size(0);

    float **d_param_ptrs;
    allocateParametersPointers(d_param_ptrs, 5,
		dmeans3D.contiguous().data<float>(),
		dopacities.contiguous().data<float>(),
		dscales.contiguous().data<float>(),
		drotations.contiguous().data<float>(),
		dshs.contiguous().data<float>());

    CudaRasterizer::Rasterizer::cat_transfer_gpu2cpu_osr_shs(
        d_param_ptrs,
        h_dparameters.contiguous().data<float>(),
        infrustum_radii_opacities_filter_indices.contiguous().data<int64_t>(),
		send2gpu_final_filter_indices.contiguous().data<int64_t>(),
        dims.contiguous().data<int>(),
        dims_presum_shift.contiguous().data<int>(),
        col2attr.contiguous().data<int>(),
        num_select,
        accum,
        false
    );

	cudaDeviceSynchronize();
	cudaFree(d_param_ptrs);
}

void SendSHS2CpuSHSBufferCUDA(
    torch::Tensor& d_dshs,
    torch::Tensor& mask_indices,
    torch::Tensor& h_dparameters,
    bool accum)
{
    int64_t num_select = mask_indices.size(0);

    CudaRasterizer::Rasterizer::transfer_cpu2gpu_shs(
        d_dshs.contiguous().data<float>(),
        mask_indices.contiguous().data<int64_t>(),
        num_select,
        h_dparameters.contiguous().data<float>(),
        accum,
        false
    );

    cudaDeviceSynchronize();
}

__global__ void transfer_shsgrad_gpu2cpu_kernel_stream(
    const float *d_parameters,
    float *h_parameters,
    const int64_t *rank2id,
    const int64_t num_select,
    bool accum
) {
    int64_t stride = gridDim.x * blockDim.x;
    int64_t total_elements = num_select * 48;

    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < total_elements; i += stride) {
        int64_t row = i / 48;
        int col = i % 48;

        int64_t offset_srce = i;
        int64_t offset_dest = rank2id[row] * 48 + col;

        if (accum) h_parameters[offset_dest] += d_parameters[offset_srce];
        else h_parameters[offset_dest] = d_parameters[offset_srce];
    }
}

void SendSHS2CpuGradBufferStreamCUDA(
    torch::Tensor& d_parameters,
    torch::Tensor& h_parameters,
    torch::Tensor& mask_indices,
    bool accum,
    int grid_size,
    int block_size)
{
    int64_t N = h_parameters.size(0);
    int64_t num_select = mask_indices.size(0);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    // const int grid_size = 32;
    // const int grid_size = 16;
    // const int grid_size = 8;
    // const int block_size = 256;

    transfer_shsgrad_gpu2cpu_kernel_stream<<<grid_size, block_size, 0, stream>>>(
        d_parameters.contiguous().data<float>(),
        h_parameters.contiguous().data<float>(),
        mask_indices.contiguous().data<int64_t>(),
        num_select,
        accum
    );
}

__global__ void transfer_H_shsgrad_gpu2cpu_kernel_stream(
    const float *d_shs,
    float *h_shs,
    const int64_t *host_indices,
    const int64_t *grad_indices,
    int64_t num_select_from_host,
    bool accum
)
{
    int64_t stride = gridDim.x * blockDim.x;
    int64_t total_elements = num_select_from_host * 48;

    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < total_elements; i += stride) {
        int64_t row = i / 48;
        int col = i % 48;

        int64_t offset_host = host_indices[row] * 48 + col;
        int64_t offset_grad = grad_indices[row] * 48 + col;

        if (accum) h_shs[offset_host] += d_shs[offset_grad];
        else h_shs[offset_host] = d_shs[offset_grad];
    }
}

__global__ void transfer_D_shsgrad_gpu2cpu_kernel_stream(
    const float *d_shs,
    float *r_shs,
    const int64_t *rtnt_indices,
    const int64_t *grad_indices,
    int64_t num_select_from_retent,
    bool accum
)
{
    int64_t stride = gridDim.x * blockDim.x;
    int64_t total_elements = num_select_from_retent * 48;

    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < total_elements; i += stride) {
        int64_t row = i / 48;
        int col = i % 48;

        int64_t offset_rtnt = rtnt_indices[row] * 48 + col;
        int64_t offset_grad = grad_indices[row] * 48 + col;

        if (accum) r_shs[offset_rtnt] += d_shs[offset_grad];
        else r_shs[offset_rtnt] = d_shs[offset_grad];
    }
}

void SendSHS2CpuGradBufferStreamRetentionCUDA(
    torch::Tensor& d_parameters, // input
    torch::Tensor& h_parameters, // output 1
    torch::Tensor& r_parameters, // output 2, on gpu, gradients retent for next iteration
    torch::Tensor& host_indices,
    torch::Tensor& rtnt_indices,
    torch::Tensor& grad_indices_to_host,
    torch::Tensor& grad_indices_to_rtnt,
    bool accum,
    int grid_size_H,
    int block_size_H,
    int grid_size_D,
    int block_size_D
)
{
    int64_t N = h_parameters.size(0); // number of all gaussians
    int64_t num_select_from_host = host_indices.size(0);
    int64_t num_select_from_retent = rtnt_indices.size(0);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    transfer_H_shsgrad_gpu2cpu_kernel_stream<<<grid_size_H, block_size_H, 0, stream>>>(
        d_parameters.contiguous().data<float>(),
        h_parameters.contiguous().data<float>(),
        host_indices.contiguous().data<int64_t>(),
        grad_indices_to_host.contiguous().data<int64_t>(),
        num_select_from_host,
        accum
    );

    transfer_D_shsgrad_gpu2cpu_kernel_stream<<<grid_size_D, block_size_D, 0, stream>>>(
        d_parameters.contiguous().data<float>(),
        r_parameters.contiguous().data<float>(),
        rtnt_indices.contiguous().data<int64_t>(),
        grad_indices_to_rtnt.contiguous().data<int64_t>(),
        num_select_from_retent,
        accum
    );
}

__global__ void transfer_H_shsgrad_gpu2cpu_kernel_stream_retention2(
    const float *d_shs,
    float *h_shs,
    const int *filter,
    int *retention_vec,
    int num_select,
    bool accum
) {
    int stride = gridDim.x * blockDim.x;
    int total_elements = num_select * 48;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total_elements; i += stride) {
        int row = filter[i / 48];
        int col = i % 48;

        if (retention_vec[row] != -2) {
            int offset_srce = i;
            int offset_dest = row * 48 + col;

            if (accum) h_shs[offset_dest] += d_shs[offset_srce];
            else h_shs[offset_dest] = d_shs[offset_srce];
        }
    }
}

__global__ void transfer_D_shsgrad_gpu2cpu_kernel_stream_retention2(
    const float *d_shs,
    float *r_shs,
    const int *filter,
    int *retention_vec,
    int num_select,
    bool accum
) {
    int stride = gridDim.x * blockDim.x;
    int total_elements = num_select * 48;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total_elements; i += stride) {
        int row = filter[i / 48];
        int col = i % 48;

        if (retention_vec[row] != -1) {
            int offset_srce = retention_vec[row] * 48 + col;
            int offset_dest = i;

            if (accum) r_shs[offset_dest] += d_shs[offset_srce];
            else r_shs[offset_dest] = d_shs[offset_srce];

            // if (col == 0) retention_vec[row] = -2;
        }
    }
}

__global__ void mark_retention_vec(
    const int *filter,
    int *retention_vec,
    int num_select
) {
    int stride = gridDim.x * blockDim.x;
    
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_select; i += stride) {
        int row = filter[i];
        if (retention_vec[row] != -1) retention_vec[row] = -2;
    }
}

void SendSHS2CpuGradBufferStreamRetention2CUDA(
    torch::Tensor& d_parameters, // input
    torch::Tensor& h_parameters, // output 1
    torch::Tensor& r_parameters, // output 2, on gpu, gradients retent for next iteration
    torch::Tensor& filter, // filter of current grads
    torch::Tensor& filter_r, // filter of retent (next grads)
    torch::Tensor& retention_vec,
    bool accum,
    int grid_size_H,
    int block_size_H,
    int grid_size_D,
    int block_size_D
)
{
    int N = h_parameters.size(0); // number of all gaussians
    int num_select = filter.size(0); // number of visible gaussians this iter
    int num_select_r = filter_r.size(0);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    transfer_D_shsgrad_gpu2cpu_kernel_stream_retention2<<<grid_size_D, block_size_D, 0, stream>>>(
        d_parameters.contiguous().data<float>(),
        r_parameters.contiguous().data<float>(),
        filter_r.contiguous().data<int>(),
        retention_vec.contiguous().data<int>(),
        num_select_r,
        accum
    );

    mark_retention_vec<<<grid_size_D, block_size_D, 0, stream>>>(
        filter_r.contiguous().data<int>(),
        retention_vec.contiguous().data<int>(),
        num_select_r
    );

    transfer_H_shsgrad_gpu2cpu_kernel_stream_retention2<<<grid_size_H, block_size_H, 0, stream>>>(
        d_parameters.contiguous().data<float>(),
        h_parameters.contiguous().data<float>(),
        filter.contiguous().data<int>(),
        retention_vec.contiguous().data<int>(),
        num_select,
        accum
    );

}

////////////////////////////////// Loss //////////////////////////////////

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
FusedLossCUDA(
    const torch::Tensor& image,
    const torch::Tensor& gt_image,
    const torch::Tensor& mask,
    const torch::Tensor& mu1,
    const torch::Tensor& mu2,
    const torch::Tensor& sigma1_sq,
    const torch::Tensor& sigma2_sq,
    const torch::Tensor& sigma12)
{
    int C = image.size(0);
    int H = image.size(1);
    int W = image.size(2);

    torch::Tensor dl1_dimage = torch::zeros({C, H, W}, image.options());
    torch::Tensor dssim_dmu1 = torch::zeros({C, H, W}, mu1.options());
    torch::Tensor dssim_dmu2 = torch::zeros({C, H, W}, mu2.options());
    torch::Tensor dssim_dsigma1_sq = torch::zeros({C, H, W}, sigma1_sq.options());
    torch::Tensor dssim_dsigma2_sq = torch::zeros({C, H, W}, sigma2_sq.options());
    torch::Tensor dssim_dsigma12 = torch::zeros({C, H, W}, sigma12.options());
    torch::Tensor l1 = torch::zeros({C, H, W}, image.options());
    torch::Tensor ssim = torch::zeros({C, H, W}, image.options());

    CudaRasterizer::Rasterizer::lossForwardBackward(
        image.contiguous().data<float>(),
        gt_image.contiguous().data<float>(),
        mask.contiguous().data<bool>(),
        mu1.contiguous().data<float>(),
        mu2.contiguous().data<float>(),
        sigma1_sq.contiguous().data<float>(),
        sigma2_sq.contiguous().data<float>(),
        sigma12.contiguous().data<float>(),
        C,
        H,
        W,
        l1.contiguous().data<float>(),
        ssim.contiguous().data<float>(),
        dl1_dimage.contiguous().data<float>(),
        dssim_dmu1.contiguous().data<float>(),
        dssim_dmu2.contiguous().data<float>(),
        dssim_dsigma1_sq.contiguous().data<float>(),
        dssim_dsigma2_sq.contiguous().data<float>(),
        dssim_dsigma12.contiguous().data<float>(),
        true
    );

    return std::make_tuple(l1, ssim, dl1_dimage, dssim_dmu1, dssim_dmu2, dssim_dsigma1_sq, dssim_dsigma2_sq, dssim_dsigma12);
}


/////////////////////////////// Preprocess ///////////////////////////////

std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
PreprocessGaussiansCUDA(
	const torch::Tensor& means3D,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const torch::Tensor& sh,
    const torch::Tensor& opacity,//3dgs' parametes.
	const float scale_modifier,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const float tan_fovx, 
	const float tan_fovy,
    const int image_height,
    const int image_width,
	const int degree,
	const torch::Tensor& campos,
	const bool prefiltered,//raster_settings
	const bool debug,
	const pybind11::dict &args) {

	if (means3D.ndimension() != 2 || means3D.size(1) != 3) {
		AT_ERROR("means3D must have dimensions (num_points, 3)");
	}

	const int P = means3D.size(0);
	const int H = image_height;
	const int W = image_width;

	// of shape (P, 2). means2D is (P, 2) in cuda. It will be converted to (P, 3) when is sent back to python to meet torch graph's requirement.
	torch::Tensor means2D = torch::full({P, 2}, 0.0, means3D.options());//TODO: what about require_grads?
	// of shape (P)
	torch::Tensor depths = torch::full({P}, 0.0, means3D.options());
	// of shape (P)
	torch::Tensor radii = torch::full({P}, 0, means3D.options().dtype(torch::kInt32));
	// of shape (P, 6)
	torch::Tensor cov3D = torch::full({P, 6}, 0.0, means3D.options());
	// of shape (P, 4)
	torch::Tensor conic_opacity = torch::full({P, 4}, 0.0, means3D.options());
	// of shape (P, 3)
	torch::Tensor rgb = torch::full({P, 3}, 0.0, means3D.options());
	// of shape (P)
	torch::Tensor clamped = torch::full({P, 3}, false, means3D.options().dtype(at::kBool));
	//TODO: compare to original GeometryState implementation, this one does not explicitly do gpu memory alignment. 
	//That may lead to problems. However, pytorch does implicit memory alignment.

	int rendered = 0;//TODO: I could compute rendered here by summing up geomState.tiles_touched. 
	if(P != 0)
	{
		int M = 0;
		if(sh.size(0) != 0)
		{
			M = sh.size(1);
		}

		rendered = CudaRasterizer::Rasterizer::preprocessForward(
			reinterpret_cast<float2*>(means2D.contiguous().data<float>()),//TODO: check whether it supports float2?
			depths.contiguous().data<float>(),
			radii.contiguous().data<int>(),
			cov3D.contiguous().data<float>(),
			reinterpret_cast<float4*>(conic_opacity.contiguous().data<float>()),
			rgb.contiguous().data<float>(),
			clamped.contiguous().data<bool>(),
			P, degree, M,
			W, H,
			means3D.contiguous().data<float>(),
			scales.contiguous().data_ptr<float>(),
			rotations.contiguous().data_ptr<float>(),
			sh.contiguous().data_ptr<float>(),
			opacity.contiguous().data<float>(), 
			scale_modifier,
			viewmatrix.contiguous().data<float>(), 
			projmatrix.contiguous().data<float>(),
			campos.contiguous().data<float>(),
			tan_fovx,
			tan_fovy,
			prefiltered,
			debug,
			args);
	}
	return std::make_tuple(rendered, means2D, depths, radii, cov3D, conic_opacity, rgb, clamped);
}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
  PreprocessGaussiansBackwardCUDA(
	const torch::Tensor& radii,
	const torch::Tensor& cov3D,
	const torch::Tensor& clamped,//the above are all per-Gaussian intemediate results.
	const torch::Tensor& means3D,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const torch::Tensor& sh,//input of this operator
	const float scale_modifier,
	const torch::Tensor& viewmatrix,
    const torch::Tensor& projmatrix,
	const float tan_fovx,
	const float tan_fovy,
    const int image_height,
    const int image_width,
	const int degree,
	const torch::Tensor& campos,//rasterization setting.
	const torch::Tensor& dL_dmeans2D,// (P, 3)
	const torch::Tensor& dL_dconic_opacity,
	const torch::Tensor& dL_dcolors,//gradients of output of this operator
	const int R,
	const bool debug,
	const pybind11::dict &args)
{
  const int P = means3D.size(0);
  const int H = image_height;
  const int W = image_width;
  
  int M = 0;
  if(sh.size(0) != 0)
  {	
	M = sh.size(1);
  }

  torch::Tensor dL_dconic = torch::zeros({P, 2, 2}, means3D.options());
  // set dL_dconic[..., 0, 0] = dL_dconic_opacity[..., 0]
  dL_dconic.select(1, 0).select(1, 0).copy_(dL_dconic_opacity.select(1, 0));// select() is kind of view, it does not allocate new memory.
  // set dL_dconic[..., 0, 1] = dL_dconic_opacity[..., 1]
  dL_dconic.select(1, 0).select(1, 1).copy_(dL_dconic_opacity.select(1, 1));
  // set dL_dconic[..., 1, 1] = dL_dconic_opacity[..., 2]
  dL_dconic.select(1, 1).select(1, 1).copy_(dL_dconic_opacity.select(1, 2));
  dL_dconic = dL_dconic.contiguous();
  //TODO: is this correct usage?

  torch::Tensor dL_dopacity = torch::zeros({P, 1}, means3D.options());
  // set dL_dopacity[..., 0] = dL_dconic_opacity[..., 3]
  dL_dopacity.select(1, 0).copy_(dL_dconic_opacity.select(1, 3));
  dL_dopacity = dL_dopacity.contiguous();

  torch::Tensor dL_dmeans3D = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dcov3D = torch::zeros({P, 6}, means3D.options());
  //dL_dcov3D is itermidiate result to compute dL_drotations and dL_dscales, do not need to return to python.
  torch::Tensor dL_dscales = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_drotations = torch::zeros({P, 4}, means3D.options());
  torch::Tensor dL_dsh = torch::zeros({P, M, 3}, means3D.options());

  if(P != 0)
  {  
	  CudaRasterizer::Rasterizer::preprocessBackward(
		radii.contiguous().data<int>(),
		cov3D.contiguous().data<float>(),
		clamped.contiguous().data<bool>(),//the above are all per-Gaussian intermediate results.
		P, degree, M, R,
		W, H, //rasterization setting.
		means3D.contiguous().data<float>(),
		scales.data_ptr<float>(),
  	    rotations.data_ptr<float>(),
		sh.contiguous().data<float>(),//input of this operator
		scale_modifier,
		viewmatrix.contiguous().data<float>(),
	    projmatrix.contiguous().data<float>(),
	    campos.contiguous().data<float>(),
	    tan_fovx,
	    tan_fovy,//rasterization setting.
	    dL_dmeans2D.contiguous().data<float>(),
	    dL_dconic.contiguous().data<float>(),
	    dL_dcolors.contiguous().data<float>(),//gradients of output of this operator
	    dL_dmeans3D.contiguous().data<float>(),
	    dL_dcov3D.contiguous().data<float>(),
	    dL_dscales.contiguous().data<float>(),
	    dL_drotations.contiguous().data<float>(),
	    dL_dsh.contiguous().data<float>(),//gradients of input of this operator
		debug,
		args);
  }

  return std::make_tuple(dL_dmeans3D, dL_dscales, dL_drotations, dL_dsh, dL_dopacity);
}


////////////////////// GetDistributionStrategy ////////////////////////

torch::Tensor GetDistributionStrategyCUDA(
    const int image_height,
    const int image_width,// image setting
	torch::Tensor& means2D,// (P, 2)
	torch::Tensor& radii,
	const bool debug,
	const pybind11::dict &args)
{
	const int P = means2D.size(0);
	const int TILE_Y = (image_height + BLOCK_Y - 1) / BLOCK_Y;
	const int TILE_X = (image_width + BLOCK_X - 1) / BLOCK_X;
	
	torch::Tensor compute_locally = torch::full({TILE_Y, TILE_X}, false, means2D.options().dtype(at::kBool).requires_grad(false));

	torch::Device device(torch::kCUDA);
	torch::TensorOptions options(torch::kByte);
	torch::Tensor distBuffer = torch::empty({0}, options.device(device));
	std::function<char*(size_t)> distFunc = resizeFunctional(distBuffer);

	if (P != 0)
	{
		CudaRasterizer::Rasterizer::getDistributionStrategy(
			distFunc,
			P,
			image_width, image_height,
			reinterpret_cast<float2*>(means2D.contiguous().data<float>()),
			radii.contiguous().data<int>(),
			compute_locally.contiguous().data<bool>(),
			debug,
			args);
	}
	return compute_locally;
}


/////////////////////////////// Render ///////////////////////////////

std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RenderGaussiansCUDA(
	const torch::Tensor& background,
    const int image_height,
    const int image_width,// image setting
	torch::Tensor& means2D,// (P, 2)
	torch::Tensor& depths,
	torch::Tensor& radii,
	torch::Tensor& conic_opacity,
	torch::Tensor& rgb,//3dgs intermediate results
	const torch::Tensor& compute_locally,
	const bool debug,
	const pybind11::dict &args)
{
  const int P = means2D.size(0);
  const int H = image_height;
  const int W = image_width;

  auto int_opts = means2D.options().dtype(torch::kInt32);
  auto float_opts = means2D.options().dtype(torch::kFloat32);

  torch::Tensor out_color = torch::full({NUM_CHANNELS, H, W}, 0.0, float_opts);

  const int TILE_Y = (H + BLOCK_Y - 1) / BLOCK_Y;
  const int TILE_X = (W + BLOCK_X - 1) / BLOCK_X;
  const int tile_num = TILE_Y * TILE_X;
  torch::Tensor n_render = torch::full({tile_num}, 0, int_opts);
  torch::Tensor n_consider = torch::full({tile_num}, 0, int_opts);
  torch::Tensor n_contrib = torch::full({tile_num}, 0, int_opts);

  torch::Device device(torch::kCUDA);
  torch::TensorOptions options(torch::kByte);
  torch::Tensor geomBuffer = torch::empty({0}, options.device(device));
  torch::Tensor binningBuffer = torch::empty({0}, options.device(device));
  torch::Tensor imgBuffer = torch::empty({0}, options.device(device));
  std::function<char*(size_t)> geomFunc = resizeFunctional(geomBuffer);
  std::function<char*(size_t)> binningFunc = resizeFunctional(binningBuffer);
  std::function<char*(size_t)> imgFunc = resizeFunctional(imgBuffer);
  
  int rendered = 0;
  if(P != 0)
  {
	  rendered = CudaRasterizer::Rasterizer::renderForward(
		geomFunc,
		binningFunc,
		imgFunc,//buffer
	    P,
		background.contiguous().data<float>(),
		W, H,//image setting
		reinterpret_cast<float2*>(means2D.contiguous().data<float>()),
		depths.contiguous().data<float>(),
		radii.contiguous().data<int>(),
		reinterpret_cast<float4*>(conic_opacity.contiguous().data<float>()),
		rgb.contiguous().data<float>(),//3dgs intermediate results
		compute_locally.contiguous().data<bool>(),
		out_color.contiguous().data<float>(),
		n_render.contiguous().data<int>(),
		n_consider.contiguous().data<int>(),
		n_contrib.contiguous().data<int>(),//output
		debug,
		args);
  }
  return std::make_tuple(rendered, out_color, n_render, n_consider, n_contrib, geomBuffer, binningBuffer, imgBuffer);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
RenderGaussiansBackwardCUDA(
 	const torch::Tensor& background,
	const int R,
	const torch::Tensor& geomBuffer,
	const torch::Tensor& binningBuffer,
	const torch::Tensor& imageBuffer,
	const torch::Tensor& compute_locally,
    const torch::Tensor& dL_dout_color,
	const torch::Tensor& means2D,// (P, 2)
	const torch::Tensor& conic_opacity,
	const torch::Tensor& rgb,
	const bool debug,
	const pybind11::dict &args)
{
  const int P = means2D.size(0);
  const int H = dL_dout_color.size(1);
  const int W = dL_dout_color.size(2);
  
  torch::Tensor dL_dmeans2D = torch::zeros({P, 3}, means2D.options());//TODO: does options for a tensor and its grad differ from each other?
  torch::Tensor dL_dcolors = torch::zeros({P, NUM_CHANNELS}, means2D.options());//if we use mixed precision, dtype in options() is different now. If we also do swapping, device could be different. 
  torch::Tensor dL_dconic = torch::zeros({P, 2, 2}, means2D.options());//The requires_grad property for the gradient tensor is typically False
  torch::Tensor dL_dopacity = torch::zeros({P, 1}, means2D.options());

  if(P != 0)
  {
	  CudaRasterizer::Rasterizer::renderBackward(
		P, R,
		background.contiguous().data<float>(),
		W, H,//rasterization settings.  
		reinterpret_cast<char*>(geomBuffer.contiguous().data_ptr()),
		reinterpret_cast<char*>(binningBuffer.contiguous().data_ptr()),
		reinterpret_cast<char*>(imageBuffer.contiguous().data_ptr()),//buffer that contains intermedia results
		compute_locally.contiguous().data<bool>(),
		dL_dout_color.contiguous().data<float>(),//gradient of output
		dL_dmeans2D.contiguous().data<float>(),
		dL_dconic.contiguous().data<float>(),
		dL_dopacity.contiguous().data<float>(),
		dL_dcolors.contiguous().data<float>(),//gradient of inputs
		reinterpret_cast<float2*>(means2D.contiguous().data<float>()),
		reinterpret_cast<float4*>(conic_opacity.contiguous().data<float>()),
		rgb.contiguous().data<float>(),
		debug,
		args);
  }

  torch::Tensor dL_dconic_opacity = torch::zeros({P, 4}, means2D.options());
  // set dL_dconic_opacity[..., 0] = dL_dconic[..., 0, 0]
  dL_dconic_opacity.select(1, 0).copy_(dL_dconic.select(1, 0).select(1, 0));
  // set dL_dconic_opacity[..., 1] = dL_dconic[..., 0, 1]
  dL_dconic_opacity.select(1, 1).copy_(dL_dconic.select(1, 0).select(1, 1));
  // set dL_dconic_opacity[..., 2] = dL_dconic[..., 1, 1]
  dL_dconic_opacity.select(1, 2).copy_(dL_dconic.select(1, 1).select(1, 1));
  // set dL_dconic_opacity[..., 3] = dL_dopacity[..., 0]
  dL_dconic_opacity.select(1, 3).copy_(dL_dopacity.select(1, 0));
  
  //TODO: in pytorch, when the reference to a tensor decreases to 0, the memory will be freed.
  //But what will happen to libtorch?
  return std::make_tuple(dL_dmeans2D, dL_dconic_opacity, dL_dcolors);
}

/////////////////////////////// Utility tools ///////////////////////////////

__global__ void getTouchedIdsBool(
	int P,
	int height,
	int width,
	int world_size,
	const float2* means2D,
	const int* radii,// NOTE: radii is not const in getRect()
	const int* dist_global_strategy,
	bool* touchedIdsBool,
	bool avoid_pixel_all2all)
{
	auto i = cg::this_grid().thread_rank();
	if (i < P)
	{
		uint2 rect_min, rect_max;
		dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);

		getRect(means2D[i], radii[i], rect_min, rect_max, tile_grid);
		
		// method 1:
		int touched_min_tile_idx = rect_min.y * tile_grid.x + rect_min.x;
		int touched_max_tile_idx = (rect_max.y - 1 ) * tile_grid.x + rect_max.x - 1;

		if ( touched_max_tile_idx < touched_min_tile_idx )
			return;
			
		for (int rk = 0; rk < world_size; rk++)
		{
			int tile_l = *(dist_global_strategy+rk);
			int tile_r = *(dist_global_strategy+rk+1);
			if (avoid_pixel_all2all) {
				// we could avoid the pixel all2all by rendering the pixels that are near border and out of border. 
				tile_l -= tile_grid.x+1;
				tile_r += tile_grid.x+1;
			}

			if (touched_max_tile_idx < tile_l || touched_min_tile_idx >= tile_r)
				continue;
			
			// TODO: If one worker's tiles are fewer than one row, then it is buggy. 
			// If we have other workload_division dimension, then we need to change this. 
			touchedIdsBool[i * world_size + rk] = true;
		}
		

		
	}
}

torch::Tensor GetLocal2jIdsBoolCUDA(
	int image_height,
	int image_width,
	int mp_rank,
	int mp_world_size,
	const torch::Tensor& means2D,
	const torch::Tensor& radii,
	const torch::Tensor& dist_global_strategy,
	const pybind11::dict &args)
{	
	const int P = means2D.size(0);
	const int H = image_height;
	const int W = image_width;
	bool avoid_pixel_all2all = args["avoid_pixel_all2all"].cast<bool>();

	torch::Tensor local2jIdsBool = torch::full({P, mp_world_size}, false, means2D.options().dtype(torch::kBool));

	getTouchedIdsBool << <(P + ONE_DIM_BLOCK_SIZE - 1) / ONE_DIM_BLOCK_SIZE, ONE_DIM_BLOCK_SIZE >> >(
		P,
		H,
		W,
		mp_world_size,
		reinterpret_cast<float2*>(means2D.contiguous().data<float>()),
		radii.contiguous().data<int>(),
		dist_global_strategy.contiguous().data<int>(),
		local2jIdsBool.contiguous().data<bool>(),
		avoid_pixel_all2all
	);

	return local2jIdsBool;
}


__global__ void getTouchedIdsBoolAdjustMode6(
	int P,
	int height,
	int width,
	int world_size,
	const float2* means2D,
	const int* radii,// NOTE: radii is not const in getRect()
	const int* rectangles,
	bool* touchedIdsBool,
	bool avoid_pixel_all2all)
{
	auto i = cg::this_grid().thread_rank();
	if (i < P)
	{
		uint2 rect_min, rect_max;
		dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);

		getRect(means2D[i], radii[i], rect_min, rect_max, tile_grid);

		for (int rk = 0; rk < world_size; rk++)
		{
			// local_tile_y_l, local_tile_y_r, local_tile_x_l, local_tile_x_r
			const int* rectangles_offset = rectangles+(rk*4);
			int local_tile_y_l = *(rectangles_offset);
			int local_tile_y_r = *(rectangles_offset+1);
			int local_tile_x_l = *(rectangles_offset+2);
			int local_tile_x_r = *(rectangles_offset+3);



			if (avoid_pixel_all2all) {
				if (local_tile_y_l>0) local_tile_y_l-=1;
				if (local_tile_x_l>0) local_tile_x_l-=1;//WERID: If local_tile_x_l changes to -1, then it gives weird behavior and I have not figure it out yet. 
				local_tile_y_r+=1;
				local_tile_x_r+=1;
			}
			if (rect_max.y <= local_tile_y_l || 
				local_tile_y_r <= rect_min.y || 
				rect_max.x <= local_tile_x_l || 
				local_tile_x_r <= rect_min.x) continue;

			touchedIdsBool[i * world_size + rk] = true;
		}
	}
}

torch::Tensor GetLocal2jIdsBoolAdjustMode6CUDA(
	int image_height,
	int image_width,
	int mp_rank,
	int mp_world_size,
	const torch::Tensor& means2D,
	const torch::Tensor& radii,
	const torch::Tensor& rectangles,
	const pybind11::dict &args)
{
	const int P = means2D.size(0);
	const int H = image_height;
	const int W = image_width;
	bool avoid_pixel_all2all = args["avoid_pixel_all2all"].cast<bool>();

	torch::Tensor local2jIdsBool = torch::full({P, mp_world_size}, false, means2D.options().dtype(torch::kBool));

	getTouchedIdsBoolAdjustMode6 << <(P + ONE_DIM_BLOCK_SIZE - 1) / ONE_DIM_BLOCK_SIZE, ONE_DIM_BLOCK_SIZE >> >(
		P,
		H,
		W,
		mp_world_size,
		reinterpret_cast<float2*>(means2D.contiguous().data<float>()),
		radii.contiguous().data<int>(),
		rectangles.contiguous().data<int>(),
		local2jIdsBool.contiguous().data<bool>(),
		avoid_pixel_all2all
	);

	return local2jIdsBool;
}





////////////////////// Image Distribution Utilities ////////////////////////


__global__ void get_touched_locally(
	const int tile_num,
	const int TILE_Y,
	const int TILE_X,
	const bool* compute_locally,
	bool* touched_locally
) {
	auto i = cg::this_grid().thread_rank();
	if (i < tile_num && compute_locally[i])
	{
		int y = i / TILE_X;
		int x = i % TILE_X;
		touched_locally[i] = true;
		const int dx[8] = {-1, -1, -1, 0, 0, 1, 1, 1};//by default, extension_distance is 1.
		const int dy[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
		for (int k = 0; k < 8; k++)
		{
			int ny = y + dy[k];
			int nx = x + dx[k];
			if (ny >= 0 && ny < TILE_Y && nx >= 0 && nx < TILE_X)
				touched_locally[ny * TILE_X + nx] = true;
		}
	}
}

torch::Tensor GetTouchedLocally(
	const torch::Tensor& compute_locally,
	const int image_height,
	const int image_width,
	const int extension_distance
) {
	const int TILE_Y = (image_height + BLOCK_Y - 1) / BLOCK_Y;
	const int TILE_X = (image_width + BLOCK_X - 1) / BLOCK_X;
	const int tile_num = TILE_Y * TILE_X;// NOTE: at most, we have 5000*5000/16/16 = 97656 tiles
	
	torch::Tensor touched_locally = torch::full({TILE_Y, TILE_X}, false, compute_locally.options());

	get_touched_locally<<< (tile_num + ONE_DIM_BLOCK_SIZE - 1) / ONE_DIM_BLOCK_SIZE, ONE_DIM_BLOCK_SIZE >>> (
		tile_num,
		TILE_Y,
		TILE_X,
		compute_locally.contiguous().data<bool>(),
		touched_locally.contiguous().data<bool>()
	);
	return touched_locally;
}


__global__ void load_image_tiles_by_pos(
	int N,
	int image_height,
	int image_width,
	int min_pixel_y,
	int min_pixel_x,
	int local_image_rect_height,
	int local_image_rect_width,
	const int64_t* all_tiles_pos,
	const float* local_image_rect,
	float* image_tiles)
{
	auto block = cg::this_thread_block();
	int i = block.group_index().x;
	int tile_pos_y = (int)all_tiles_pos[ i * 2 ];
	int tile_pos_x = (int)all_tiles_pos[ i * 2 + 1 ];

	int image_x = tile_pos_x * BLOCK_X + block.thread_index().x;
	int image_y = tile_pos_y * BLOCK_Y + block.thread_index().y;

	int image_tiles_offset = i * 3 * BLOCK_X * BLOCK_Y + block.thread_rank();
	int tile_pixels_num = BLOCK_X * BLOCK_Y;

	if (image_x < image_width && image_y < image_height)
	{
		int local_image_rect_x = image_x - min_pixel_x;
		int local_image_rect_y = image_y - min_pixel_y;
		int local_image_rect_offset = local_image_rect_y * local_image_rect_width + local_image_rect_x;
		int local_image_rect_pixels_num = local_image_rect_height * local_image_rect_width;

		image_tiles[image_tiles_offset] = local_image_rect[local_image_rect_offset];
		image_tiles[image_tiles_offset + tile_pixels_num] = local_image_rect[local_image_rect_offset + local_image_rect_pixels_num];
		image_tiles[image_tiles_offset + 2 * tile_pixels_num] = local_image_rect[local_image_rect_offset + 2 * local_image_rect_pixels_num];
	}
	else
	{
		image_tiles[image_tiles_offset] = 0.0;
		image_tiles[image_tiles_offset + tile_pixels_num] = 0.0;
		image_tiles[image_tiles_offset + 2 * tile_pixels_num] = 0.0;
	}
}

__global__ void set_image_tiles_by_pos(
	int N,
	int image_height,
	int image_width,
	int min_pixel_y,
	int min_pixel_x,
	int local_image_rect_height,
	int local_image_rect_width,
	const int64_t* all_tiles_pos,
	float* local_image_rect,
	const float* image_tiles)
{
	auto block = cg::this_thread_block();
	int i = block.group_index().x;
	int tile_pos_y = (int)all_tiles_pos[ i * 2 ];
	int tile_pos_x = (int)all_tiles_pos[ i * 2 + 1 ];

	int image_x = tile_pos_x * BLOCK_X + block.thread_index().x;
	int image_y = tile_pos_y * BLOCK_Y + block.thread_index().y;

	int image_tiles_offset = i * 3 * BLOCK_X * BLOCK_Y + block.thread_rank();
	if (image_x < image_width && image_y < image_height)
	{
		int local_image_rect_x = image_x - min_pixel_x;
		int local_image_rect_y = image_y - min_pixel_y;
		int local_image_rect_offset = local_image_rect_y * local_image_rect_width + local_image_rect_x;
		int local_image_rect_pixels_num = local_image_rect_height * local_image_rect_width;
		int tile_pixels_num = BLOCK_X * BLOCK_Y;

		local_image_rect[local_image_rect_offset] = image_tiles[image_tiles_offset];
		local_image_rect[local_image_rect_offset + local_image_rect_pixels_num] = image_tiles[image_tiles_offset + tile_pixels_num];
		local_image_rect[local_image_rect_offset + 2 * local_image_rect_pixels_num] = image_tiles[image_tiles_offset + 2 * tile_pixels_num];
	}
}

torch::Tensor LoadImageTilesByPos(
	const torch::Tensor& local_image_rect,
	const torch::Tensor& all_tiles_pos,
	int image_height,
	int image_width,
	int min_pixel_y,
	int min_pixel_x,
	int local_image_rect_height,
	int local_image_rect_width)
{
	const int N = all_tiles_pos.size(0);
	dim3 tile_grid(N, 1, 1);
	dim3 block(BLOCK_X, BLOCK_Y, 1);

	torch::Tensor image_tiles = torch::full({N, 3, BLOCK_Y, BLOCK_X}, 0.0, local_image_rect.options());
	// if image.options() requires_grad, then image_tiles.options() requires_grad should also requires_grad.

	load_image_tiles_by_pos<<< tile_grid, block >>>(
		N,
		image_height,
		image_width,
		min_pixel_y,
		min_pixel_x,
		local_image_rect_height,
		local_image_rect_width,
		all_tiles_pos.contiguous().data<int64_t>(),
		local_image_rect.contiguous().data<float>(),
		image_tiles.contiguous().data<float>()
	);
	return image_tiles;
}

torch::Tensor SetImageTilesByPos(
	const torch::Tensor& all_tiles_pos,
	const torch::Tensor& image_tiles,
	int image_height,
	int image_width,
	int min_pixel_y,
	int min_pixel_x,
	int local_image_rect_height,
	int local_image_rect_width)
{
	const int N = all_tiles_pos.size(0);
	dim3 tile_grid(N, 1, 1);
	dim3 block(BLOCK_X, BLOCK_Y, 1);

	torch::Tensor local_image_rect = torch::full({3, local_image_rect_height, local_image_rect_width}, 0.0, image_tiles.options());

	set_image_tiles_by_pos <<< tile_grid, block >>>(
		N,
		image_height,
		image_width,
		min_pixel_y,
		min_pixel_x,
		local_image_rect_height,
		local_image_rect_width,
		all_tiles_pos.contiguous().data<int64_t>(),
		local_image_rect.contiguous().data<float>(),
		image_tiles.contiguous().data<float>()
	);
	return local_image_rect;
}


__global__ void get_pixels_compute_locally_and_in_rect(
	int image_height,
	int image_width,
	int local_image_height,
	int local_image_width,
	int min_pixel_y,
	int min_pixel_x,
	const bool* compute_locally,
	bool* pixels_compute_locally_and_in_rect)
{
	auto block = cg::this_thread_block();
	int local_pixel_x = block.group_index().x * BLOCK_X + block.thread_index().x;
	int local_pixel_y = block.group_index().y * BLOCK_Y + block.thread_index().y;

	if (local_pixel_x < local_image_width && local_pixel_y < local_image_height)
	{
		int global_pixel_x = local_pixel_x + min_pixel_x;
		int global_pixel_y = local_pixel_y + min_pixel_y;
		int global_tile_x = global_pixel_x / BLOCK_X;
		int global_tile_y = global_pixel_y / BLOCK_Y;
		int TILE_X = (image_width + BLOCK_X - 1) / BLOCK_X;
		pixels_compute_locally_and_in_rect[local_pixel_y * local_image_width + local_pixel_x] = compute_locally[global_tile_y * TILE_X + global_tile_x];
	}
}

torch::Tensor GetPixelsComputeLocallyAndInRect(
	const torch::Tensor& compute_locally,
	int image_height,
	int image_width,
	int min_pixel_y,
	int max_pixel_y,
	int min_pixel_x,
	int max_pixel_x)
{
	int local_image_height = max_pixel_y - min_pixel_y;
	int local_image_width = max_pixel_x - min_pixel_x;
	const int TILE_Y = (local_image_height + BLOCK_Y - 1) / BLOCK_Y;
	const int TILE_X = (local_image_width + BLOCK_X - 1) / BLOCK_X;

	dim3 tile_grid(TILE_X, TILE_Y, 1);
	dim3 block(BLOCK_X, BLOCK_Y, 1);
	
	torch::Tensor pixels_compute_locally_and_in_rect = torch::full({max_pixel_y - min_pixel_y, max_pixel_x - min_pixel_x}, false, compute_locally.options().dtype(at::kBool));

	get_pixels_compute_locally_and_in_rect << < tile_grid, block >> > (
		image_height,
		image_width,	
		local_image_height,
		local_image_width,
		min_pixel_y,
		min_pixel_x,
		compute_locally.contiguous().data<bool>(),
		pixels_compute_locally_and_in_rect.contiguous().data<bool>()
	);
	return pixels_compute_locally_and_in_rect;
}

std::tuple<int, int, int> GetBlockXY()
{
	return std::make_tuple(BLOCK_X, BLOCK_Y, ONE_DIM_BLOCK_SIZE);
}

__global__ void set_signal_kernel(
    int* signal_tensor,
    int microbatch_idx,
    int signal)
{
    __threadfence_system();
    signal_tensor[microbatch_idx] = signal;
    __threadfence_system();
}

void SetSignal(torch::Tensor& signal_tensor, int microbatch_idx, int signal)
{
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    set_signal_kernel<<<1, 1, 0, stream>>>(
        reinterpret_cast<int*>(signal_tensor.contiguous().data_ptr()),
        microbatch_idx,
        signal
    );

}
