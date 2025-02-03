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

#include <torch/extension.h>
#include "rasterize_points.h"
#include "cuda_rasterizer/config.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("mark_visible", &markVisible);
  m.def("preprocess_gaussians", &PreprocessGaussiansCUDA);
  m.def("preprocess_gaussians_backward", &PreprocessGaussiansBackwardCUDA);
  m.def("get_distribution_strategy", &GetDistributionStrategyCUDA);
  m.def("render_gaussians", &RenderGaussiansCUDA);
  m.def("render_gaussians_backward", &RenderGaussiansBackwardCUDA);
  m.def("get_local2j_ids_bool", &GetLocal2jIdsBoolCUDA);
  m.def("get_local2j_ids_bool_adjust_mode6", &GetLocal2jIdsBoolAdjustMode6CUDA);

  // Image Distribution Utilities
  m.def("get_touched_locally", &GetTouchedLocally);
  m.def("load_image_tiles_by_pos", &LoadImageTilesByPos);
  m.def("set_image_tiles_by_pos", &SetImageTilesByPos);
  m.def("get_pixels_compute_locally_and_in_rect", &GetPixelsComputeLocallyAndInRect);

  m.def("get_block_XY", &GetBlockXY);

  m.def("get_send2gpu", &GetSend2GpuCUDA);
  m.def("send2gpu", &Send2GpuCUDA);
  m.def("send_cat2gpu", &SendCat2GpuCUDA);
  m.def("send_cat2gpu_xyz", &SendCat2GpuXYZCUDA);
  m.def("send_cat2gpu_osr", &SendCat2GpuOSRCUDA);
  m.def("send_cat2gpu_shs", &SendCat2GpuSHSCUDA);
  m.def("send_shs2gpu_shs", &SendSHS2GpuSHSCUDA);
  m.def("send_shs2gpu_stream", &SendSHS2GpuStreamCUDA);
  m.def("send_shs2gpu_stream_retention", &SendSHS2GpuStreamRetentionCUDA);
  m.def("send_shs2gpu_stream_retention2", &SendSHS2GpuStreamRetention2CUDA);
  m.def("send_shs2gpu_stream_retention2_64", &SendSHS2GpuStreamRetention2_64CUDA);
  m.def("send_cat2gpu_buffer", &SendCat2GpuBufferCUDA);
  m.def("send2cpu_deprecated", &Send2CpuCUDA_deprecated);
  m.def("send2cpu", &Send2CpuCUDA);
  m.def("send2cpu_cat", &Send2CpuCatCUDA);
  m.def("send2cpu_cat_buffer", &Send2CpuCatBufferCUDA);
  m.def("send2cpu_cat_buffer_osr_shs", &Send2CpuCatBufferOSRSHSCUDA);
  m.def("send_shs2cpu_shs_buffer", &SendSHS2CpuSHSBufferCUDA);
  m.def("send_shs2cpu_grad_buffer_stream", &SendSHS2CpuGradBufferStreamCUDA);
  m.def("send_shs2cpu_grad_buffer_stream_retention", &SendSHS2CpuGradBufferStreamRetentionCUDA);
  m.def("send_shs2cpu_grad_buffer_stream_retention2_64", &SendSHS2CpuGradBufferStreamRetention2_64CUDA);
  m.def("fused_loss", &FusedLossCUDA);

  m.def("set_signal", &SetSignal);
}