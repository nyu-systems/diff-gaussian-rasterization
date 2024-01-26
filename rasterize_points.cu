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

#include <cooperative_groups.h>
namespace cg = cooperative_groups;

std::function<char*(size_t N)> resizeFunctional(torch::Tensor& t) {
    auto lambda = [&t](size_t N) {
        t.resize_({(long long)N});
		return reinterpret_cast<char*>(t.contiguous().data_ptr());
    };
    return lambda;
}

std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansCUDA(
	const torch::Tensor& background,
	const torch::Tensor& means3D,
    const torch::Tensor& colors,
    const torch::Tensor& opacity,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const float tan_fovx, 
	const float tan_fovy,
    const int image_height,
    const int image_width,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const bool prefiltered,
	const bool debug,
	const pybind11::dict &args)
{
  if (means3D.ndimension() != 2 || means3D.size(1) != 3) {
    AT_ERROR("means3D must have dimensions (num_points, 3)");
  }
  
  const int P = means3D.size(0);
  const int H = image_height;
  const int W = image_width;

  auto int_opts = means3D.options().dtype(torch::kInt32);
  auto float_opts = means3D.options().dtype(torch::kFloat32);

  torch::Tensor out_color = torch::full({NUM_CHANNELS, H, W}, 0.0, float_opts);
  torch::Tensor radii = torch::full({P}, 0, means3D.options().dtype(torch::kInt32));

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
  torch::Tensor distBuffer = torch::empty({0}, options.device(device));
  std::function<char*(size_t)> geomFunc = resizeFunctional(geomBuffer);
  std::function<char*(size_t)> binningFunc = resizeFunctional(binningBuffer);
  std::function<char*(size_t)> imgFunc = resizeFunctional(imgBuffer);
  std::function<char*(size_t)> distFunc = resizeFunctional(distBuffer);
  
  int rendered = 0;
  if(P != 0)
  {
	  int M = 0;
	  if(sh.size(0) != 0)
	  {
		M = sh.size(1);
      }

	  rendered = CudaRasterizer::Rasterizer::forward(
	    geomFunc,
		binningFunc,
		imgFunc,
		distFunc,
	    P, degree, M,
		background.contiguous().data<float>(),
		W, H,
		means3D.contiguous().data<float>(),
		sh.contiguous().data_ptr<float>(),
		colors.contiguous().data<float>(), 
		opacity.contiguous().data<float>(), 
		scales.contiguous().data_ptr<float>(),
		scale_modifier,
		rotations.contiguous().data_ptr<float>(),
		cov3D_precomp.contiguous().data<float>(), 
		viewmatrix.contiguous().data<float>(), 
		projmatrix.contiguous().data<float>(),
		campos.contiguous().data<float>(),
		tan_fovx,
		tan_fovy,
		prefiltered,
		out_color.contiguous().data<float>(),
		radii.contiguous().data<int>(),
		n_render.contiguous().data<int>(),
		n_consider.contiguous().data<int>(),
		n_contrib.contiguous().data<int>(),
		debug,
		args);
  }
  return std::make_tuple(rendered, out_color, radii, n_render, n_consider, n_contrib, geomBuffer, binningBuffer, imgBuffer, distBuffer);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
 RasterizeGaussiansBackwardCUDA(
 	const torch::Tensor& background,
	const torch::Tensor& means3D,
	const torch::Tensor& radii,
    const torch::Tensor& colors,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& viewmatrix,
    const torch::Tensor& projmatrix,
	const float tan_fovx,
	const float tan_fovy,
    const torch::Tensor& dL_dout_color,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const torch::Tensor& geomBuffer,
	const int R,
	const torch::Tensor& binningBuffer,
	const torch::Tensor& imageBuffer,
	const torch::Tensor& distBuffer,
	const bool debug,
	const pybind11::dict &args) 
{
  const int P = means3D.size(0);
  const int H = dL_dout_color.size(1);
  const int W = dL_dout_color.size(2);
  
  int M = 0;
  if(sh.size(0) != 0)
  {	
	M = sh.size(1);
  }

  torch::Tensor dL_dmeans3D = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dmeans2D = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dcolors = torch::zeros({P, NUM_CHANNELS}, means3D.options());
  torch::Tensor dL_dconic = torch::zeros({P, 2, 2}, means3D.options());
  torch::Tensor dL_dopacity = torch::zeros({P, 1}, means3D.options());
  torch::Tensor dL_dcov3D = torch::zeros({P, 6}, means3D.options());
  torch::Tensor dL_dsh = torch::zeros({P, M, 3}, means3D.options());
  torch::Tensor dL_dscales = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_drotations = torch::zeros({P, 4}, means3D.options());
  
  if(P != 0)
  {  
	  CudaRasterizer::Rasterizer::backward(P, degree, M, R,
	  background.contiguous().data<float>(),
	  W, H, 
	  means3D.contiguous().data<float>(),
	  sh.contiguous().data<float>(),
	  colors.contiguous().data<float>(),
	  scales.data_ptr<float>(),
	  scale_modifier,
	  rotations.data_ptr<float>(),
	  cov3D_precomp.contiguous().data<float>(),
	  viewmatrix.contiguous().data<float>(),
	  projmatrix.contiguous().data<float>(),
	  campos.contiguous().data<float>(),
	  tan_fovx,
	  tan_fovy,
	  radii.contiguous().data<int>(),
	  reinterpret_cast<char*>(geomBuffer.contiguous().data_ptr()),
	  reinterpret_cast<char*>(binningBuffer.contiguous().data_ptr()),
	  reinterpret_cast<char*>(imageBuffer.contiguous().data_ptr()),
	  reinterpret_cast<char*>(distBuffer.contiguous().data_ptr()),
	  dL_dout_color.contiguous().data<float>(),
	  dL_dmeans2D.contiguous().data<float>(),
	  dL_dconic.contiguous().data<float>(),  
	  dL_dopacity.contiguous().data<float>(),
	  dL_dcolors.contiguous().data<float>(),
	  dL_dmeans3D.contiguous().data<float>(),
	  dL_dcov3D.contiguous().data<float>(),
	  dL_dsh.contiguous().data<float>(),
	  dL_dscales.contiguous().data<float>(),
	  dL_drotations.contiguous().data<float>(),
	  debug,
	  args);
  }

  return std::make_tuple(dL_dmeans2D, dL_dcolors, dL_dopacity, dL_dmeans3D, dL_dcov3D, dL_dsh, dL_dscales, dL_drotations);
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


/////////////////////////////// Preprocess ///////////////////////////////

std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
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
	// of shape (P)
	torch::Tensor tiles_touched = torch::full({P}, 0, means3D.options().dtype(torch::kInt32));
	//TODO: check uint or int? the internal implement is using unit32_t, it is not compatible for now. 
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
			reinterpret_cast<uint32_t*>(tiles_touched.contiguous().data<int>()),//TODO: there could be a problem when cast to uint32_t from int.
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
	return std::make_tuple(rendered, means2D, depths, radii, cov3D, conic_opacity, rgb, clamped, tiles_touched);
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

/////////////////////////////// Render ///////////////////////////////

std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RenderGaussiansCUDA(
	const torch::Tensor& background,
    const int image_height,
    const int image_width,// image setting
	torch::Tensor& means2D,// (P, 2)
	torch::Tensor& depths,
	torch::Tensor& radii,
	torch::Tensor& conic_opacity,
	torch::Tensor& rgb,
	torch::Tensor& tiles_touched,//3dgs intermediate results
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
  torch::Tensor distBuffer = torch::empty({0}, options.device(device));
  std::function<char*(size_t)> geomFunc = resizeFunctional(geomBuffer);
  std::function<char*(size_t)> binningFunc = resizeFunctional(binningBuffer);
  std::function<char*(size_t)> imgFunc = resizeFunctional(imgBuffer);
  std::function<char*(size_t)> distFunc = resizeFunctional(distBuffer);
  
  int rendered = 0;
  if(P != 0)
  {
	  rendered = CudaRasterizer::Rasterizer::renderForward(
		geomFunc,
		binningFunc,
		imgFunc,
		distFunc,//buffer
	    P,
		background.contiguous().data<float>(),
		W, H,//image setting
		reinterpret_cast<float2*>(means2D.contiguous().data<float>()),
		depths.contiguous().data<float>(),
		radii.contiguous().data<int>(),
		reinterpret_cast<float4*>(conic_opacity.contiguous().data<float>()),
		rgb.contiguous().data<float>(),
		reinterpret_cast<uint32_t*>(tiles_touched.contiguous().data<int>()),//3dgs intermediate results
		out_color.contiguous().data<float>(),
		n_render.contiguous().data<int>(),
		n_consider.contiguous().data<int>(),
		n_contrib.contiguous().data<int>(),//output
		debug,
		args);
  }
  return std::make_tuple(rendered, out_color, n_render, n_consider, n_contrib, geomBuffer, binningBuffer, imgBuffer, distBuffer);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
RenderGaussiansBackwardCUDA(
 	const torch::Tensor& background,
	const int R,
	const torch::Tensor& geomBuffer,
	const torch::Tensor& binningBuffer,
	const torch::Tensor& imageBuffer,
	const torch::Tensor& distBuffer,
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
		reinterpret_cast<char*>(imageBuffer.contiguous().data_ptr()),
		reinterpret_cast<char*>(distBuffer.contiguous().data_ptr()),//buffer that contains intermedia results
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
	bool* touchedIdsBool)
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
	int local_rank,
	int world_size,
	const torch::Tensor& means2D,
	const torch::Tensor& radii,
	const torch::Tensor& dist_global_strategy,
	const pybind11::dict &args)
{	
	const int P = means2D.size(0);
	const int H = image_height;
	const int W = image_width;

	torch::Tensor local2jIdsBool = torch::full({P, world_size}, false, means2D.options().dtype(torch::kBool));

	getTouchedIdsBool << <(P + 255) / 256, 256 >> >(
		P,
		H,
		W,
		world_size,
		reinterpret_cast<float2*>(means2D.contiguous().data<float>()),
		radii.contiguous().data<int>(),
		dist_global_strategy.contiguous().data<int>(),
		local2jIdsBool.contiguous().data<bool>()
	);

	return local2jIdsBool;
}