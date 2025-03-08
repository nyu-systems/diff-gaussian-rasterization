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

#pragma once
#include <torch/extension.h>
#include <cstdio>
#include <tuple>
#include <string>
	
		
torch::Tensor markVisible(
		torch::Tensor& means3D,
		torch::Tensor& viewmatrix,
		torch::Tensor& projmatrix);

torch::Tensor GetSend2GpuCUDA(
    torch::Tensor& means3D,
    torch::Tensor& viewmatrix,
    torch::Tensor& projmatrix);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
Send2GpuCUDA(
    torch::Tensor& opacities,
    torch::Tensor& scales,
    torch::Tensor& rotations,
    torch::Tensor& features_dc,
    torch::Tensor& features_rest,
    torch::Tensor& mask);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
SendCat2GpuCUDA(
    torch::Tensor& parameters,
    torch::Tensor& mask,
    torch::Tensor& dims,
    torch::Tensor& dims_presum_shift,
    torch::Tensor& col2attr);

torch::Tensor SendCat2GpuXYZCUDA(
    torch::Tensor& parameters);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
SendCat2GpuOSRCUDA(
    torch::Tensor& parameters,
    torch::Tensor& mask,
    torch::Tensor& mask_indices,
	torch::Tensor& dims,
    torch::Tensor& dims_presum_shift,
    torch::Tensor& col2attr);

torch::Tensor SendCat2GpuSHSCUDA(
    torch::Tensor& parameters,
    torch::Tensor& mask_indices);

torch::Tensor SendSHS2GpuSHSCUDA(
    torch::Tensor& parameters,
    torch::Tensor& mask_indices);

void SendSHS2GpuStreamCUDA(
    torch::Tensor& d_parameters,
    torch::Tensor& h_parameters,
    torch::Tensor& mask_indices,
    int grid_size,
    int block_size);

void SendSHS2GpuStreamRetentionCUDA(
    torch::Tensor& d_parameters,
    torch::Tensor& h_parameters,
    torch::Tensor& r_parameters,
    torch::Tensor& host_indices,
    torch::Tensor& rtnt_indices,
    torch::Tensor& param_indices_from_host,
    torch::Tensor& param_indices_from_rtnt,
    int grid_size_H,
    int block_size_H,
    int grid_size_D,
    int block_size_D);

void SendSHS2GpuStreamRetention2CUDA(
    torch::Tensor& d_parameters,
    torch::Tensor& h_parameters,
    torch::Tensor& r_parameters,
    torch::Tensor& filter,
    torch::Tensor& retention_vec,
    int grid_size_H,
    int block_size_H,
    int grid_size_D,
    int block_size_D);

void SendSHS2GpuStreamRetention2_64CUDA(
    torch::Tensor& d_parameters,
    torch::Tensor& h_parameters,
    torch::Tensor& r_parameters,
    torch::Tensor& filter,
    torch::Tensor& retention_vec,
    int grid_size_H,
    int block_size_H,
    int grid_size_D,
    int block_size_D);

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
    torch::Tensor& d_features_rest);

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
    bool accum);

void SendSHS2CpuSHSBufferCUDA(
    torch::Tensor& d_dshs,
    torch::Tensor& mask_indices,
    torch::Tensor& h_dparameters,
    bool accum);

void SendSHS2CpuGradBufferStreamCUDA(
    torch::Tensor& d_parameters,
    torch::Tensor& h_parameters,
    torch::Tensor& mask_indices,
    bool accum,
    int grid_size,
    int block_size);

void SendSHS2CpuGradBufferStreamRetentionCUDA(
    torch::Tensor& d_parameters,
    torch::Tensor& h_parameters,
    torch::Tensor& r_parameters,
    torch::Tensor& host_indices,
    torch::Tensor& rtnt_indices,
    torch::Tensor& grad_indices_to_host,
    torch::Tensor& grad_indices_to_rtnt,
    bool accum,
    int grid_size_H,
    int block_size_H,
    int grid_size_D,
    int block_size_D);

void SendSHS2CpuGradBufferStreamRetention2_64CUDA(
    torch::Tensor& d_parameters,
    torch::Tensor& h_parameters,
    torch::Tensor& r_parameters,
    torch::Tensor& filter,
    torch::Tensor& filter_r,
    torch::Tensor& retention_vec,
    bool accum,
    int grid_size_H,
    int block_size_H,
    int grid_size_D,
    int block_size_D);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
Send2CpuCUDA_deprecated(
    torch::Tensor& dmeans3D,
    torch::Tensor& dopacities,
    torch::Tensor& dscales,
    torch::Tensor& drotations,
    torch::Tensor& dfeatures_dc,
    torch::Tensor& dfeatures_rest,
    torch::Tensor& mask);

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
    torch::Tensor& h_dfeatures_rest);

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
    torch::Tensor& col2attr);

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
    torch::Tensor& h_dparameters);


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
    const torch::Tensor& sigma12);



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
	const pybind11::dict &args);

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
	const torch::Tensor& dL_dmeans2D,
	const torch::Tensor& dL_dconic_opacity,
	const torch::Tensor& dL_dcolors,//gradients of output of this operator
	const int R,
	const bool debug,
	const pybind11::dict &args);


////////////////////// GetDistributionStrategy ////////////////////////

torch::Tensor GetDistributionStrategyCUDA(
    const int image_height,
    const int image_width,// image setting
	torch::Tensor& means2D,// (P, 2)
	torch::Tensor& radii,
	const bool debug,
	const pybind11::dict &args);




////////////////////// Image Distribution Utilities ////////////////////////

torch::Tensor GetTouchedLocally(
	const torch::Tensor& compute_locally,
	const int image_height,
	const int image_width,
	const int extension_distance
);

torch::Tensor LoadImageTilesByPos(
	const torch::Tensor& local_image_rect,
	const torch::Tensor& all_tiles_pos,
	int image_height,
	int image_width,
	int min_pixel_y,
	int min_pixel_x,
	int local_image_rect_height,
	int local_image_rect_width);

torch::Tensor SetImageTilesByPos(
	const torch::Tensor& all_tiles_pos,
	const torch::Tensor& image_tiles,
	int image_height,
	int image_width,
	int min_pixel_y,
	int min_pixel_x,
	int local_image_rect_height,
	int local_image_rect_width);

torch::Tensor GetPixelsComputeLocallyAndInRect(
	const torch::Tensor& compute_locally,
	int image_height,
	int image_width,
	int min_pixel_y,
	int max_pixel_y,
	int min_pixel_x,
	int max_pixel_x);




/////////////////////////////// Render ///////////////////////////////


std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RenderGaussiansCUDA(
	const torch::Tensor& background,
    const int image_height,
    const int image_width,// image setting
	torch::Tensor& means2D,
	torch::Tensor& depths,
	torch::Tensor& radii,
	torch::Tensor& conic_opacity,
	torch::Tensor& rgb,//3dgs intermediate results
	const torch::Tensor& compute_locally,
	const bool debug,
	const pybind11::dict &args);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
RenderGaussiansBackwardCUDA(
 	const torch::Tensor& background,
	const int R,
	const torch::Tensor& geomBuffer,
	const torch::Tensor& binningBuffer,
	const torch::Tensor& imageBuffer,
	const torch::Tensor& compute_locally,
    const torch::Tensor& dL_dout_color,
	const torch::Tensor& means2D,
	const torch::Tensor& conic_opacity,
	const torch::Tensor& rgb,
	const bool debug,
	const pybind11::dict &args);



/////////////////////////////// Utility tools ///////////////////////////////



torch::Tensor GetLocal2jIdsBoolCUDA(
	int image_height,
	int image_width,
	int mp_rank,
	int mp_world_size,
	const torch::Tensor& means2D,
	const torch::Tensor& radii,
	const torch::Tensor& dist_global_strategy,
	const pybind11::dict &args);

torch::Tensor GetLocal2jIdsBoolAdjustMode6CUDA(
	int image_height,
	int image_width,
	int mp_rank,
	int mp_world_size,
	const torch::Tensor& means2D,
	const torch::Tensor& radii,
	const torch::Tensor& rectangles,
	const pybind11::dict &args);


std::tuple<int, int, int> GetBlockXY();

void SetSignal(torch::Tensor& signal_tensor, int microbatch_idx, int signal);

void MapBitAndVec(
    torch::Tensor& bitmap,
    torch::Tensor& bitvec,
    int bit_offset, // which bit to extract: lsb=0, msb=63
    torch::Tensor& outvec
);

void GenerateHdg(
    torch::Tensor& bitmap,
    int this_bit_offset,
    int next_bit_offset,
    torch::Tensor& hdg_vec
);

void ExtractResetBitmap(
    torch::Tensor& reset_col_gathered,
    torch::Tensor& outmap
);

void ExtractFFS(
    torch::Tensor &input,
    torch::Tensor &output
);

void ScatterToBit(
    torch::Tensor& bitmap,
    torch::Tensor& filter,
    int bit
);

void ComputeCntH(
    torch::Tensor &bitmap,
    torch::Tensor &tmp_buffer,
    int grid_size,
    int blk_size
);