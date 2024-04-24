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




/////////////////////////////// Loss ///////////////////////////////

std::tuple<float, torch::Tensor>
FusedL1LossCUDA(
  torch::Tensor& image,
  torch::Tensor& gt_image,
  torch::Tensor& mask,
  float lambda_dssim
);

/////////////////////////////// Utility tools ///////////////////////////////



torch::Tensor GetLocal2jIdsBoolCUDA(
	int image_height,
	int image_width,
	int local_rank,
	int world_size,
	const torch::Tensor& means2D,
	const torch::Tensor& radii,
	const torch::Tensor& dist_global_strategy,
	const pybind11::dict &args);

std::tuple<int, int, int> GetBlockXY();