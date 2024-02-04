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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rasterize_gaussians", &RasterizeGaussiansCUDA);
  m.def("rasterize_gaussians_backward", &RasterizeGaussiansBackwardCUDA);
  m.def("mark_visible", &markVisible);
  m.def("preprocess_gaussians", &PreprocessGaussiansCUDA);
  m.def("preprocess_gaussians_backward", &PreprocessGaussiansBackwardCUDA);
  m.def("get_distribution_strategy", &GetDistributionStrategyCUDA);
  m.def("render_gaussians", &RenderGaussiansCUDA);
  m.def("render_gaussians_backward", &RenderGaussiansBackwardCUDA);
  m.def("get_local2j_ids_bool", &GetLocal2jIdsBoolCUDA);

  // Image Distribution Utilities
  m.def("get_touched_locally", &GetTouchedLocally);
  m.def("load_image_tiles_by_pos", &LoadImageTilesByPos);
  m.def("set_image_tiles_by_pos", &SetImageTilesByPos);
  m.def("get_pixels_compute_locally_and_in_rect", &GetPixelsComputeLocallyAndInRect);
}