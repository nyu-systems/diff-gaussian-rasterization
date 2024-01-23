#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from typing import NamedTuple
import torch.nn as nn
import torch
from . import _C

def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple]
    return tuple(copied_tensors)

def rasterize_gaussians(
    means3D,
    means2D,
    sh,
    colors_precomp,
    opacities,
    scales,
    rotations,
    cov3Ds_precomp,
    raster_settings,
    cuda_args,
):
    return _RasterizeGaussians.apply(
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
        cuda_args,
    )

class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
        cuda_args,
    ):

        # Restructure arguments the way that the C++ lib expects them
        args = (
            raster_settings.bg, 
            means3D,
            colors_precomp,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            sh,
            raster_settings.sh_degree,
            raster_settings.campos,
            raster_settings.prefiltered,
            raster_settings.debug,
            cuda_args
        )

        # Invoke C++/CUDA rasterizer
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:  # TODO: n_contrib return here. 
                num_rendered, color, radii, n_render, n_consider, n_contrib, geomBuffer, binningBuffer, imgBuffer, distBuffer = _C.rasterize_gaussians(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_fw.dump")
                print("\nAn error occured in forward. Please forward snapshot_fw.dump for debugging.")
                raise ex
        else:
            num_rendered, color, radii, n_render, n_consider, n_contrib, geomBuffer, binningBuffer, imgBuffer, distBuffer = _C.rasterize_gaussians(*args)

        # Keep relevant tensors for backward
        ctx.raster_settings = raster_settings
        ctx.cuda_args = cuda_args
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer, distBuffer)
        return color, radii, n_render, n_consider, n_contrib

    @staticmethod
    def backward(ctx, grad_out_color, grad_out_radii, grad_out_n_render, grad_out_n_consider, grad_out_n_contrib):
        # grad_out_radii, grad_out_n_render, grad_out_n_consider, grad_out_n_contrib should be all None. 

        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        cuda_args = ctx.cuda_args
        colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer, distBuffer = ctx.saved_tensors

        # Restructure args as C++ method expects them
        args = (raster_settings.bg,
                means3D, 
                radii, 
                colors_precomp, 
                scales, 
                rotations, 
                raster_settings.scale_modifier, 
                cov3Ds_precomp, 
                raster_settings.viewmatrix, 
                raster_settings.projmatrix, 
                raster_settings.tanfovx, 
                raster_settings.tanfovy, 
                grad_out_color, 
                sh, 
                raster_settings.sh_degree, 
                raster_settings.campos,
                geomBuffer,
                num_rendered,
                binningBuffer,
                imgBuffer,
                distBuffer,
                raster_settings.debug,
                cuda_args)

        # Compute gradients for relevant tensors by invoking backward method
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations = _C.rasterize_gaussians_backward(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_bw.dump")
                print("\nAn error occured in backward. Writing snapshot_bw.dump for debugging.\n")
                raise ex
        else:
             grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations = _C.rasterize_gaussians_backward(*args)

        grads = (
            grad_means3D,
            grad_means2D,
            grad_sh,
            grad_colors_precomp,
            grad_opacities,
            grad_scales,
            grad_rotations,
            grad_cov3Ds_precomp,
            None,
            None # this is for cuda_args
        )

        return grads




########################### Preprocess ###########################



def preprocess_gaussians(
    means3D,
    scales,
    rotations,
    sh,
    opacities,
    raster_settings,
    cuda_args,
):
    return _PreprocessGaussians.apply(
        means3D,
        scales,
        rotations,
        sh,
        opacities,
        raster_settings,
        cuda_args,
    )

class _PreprocessGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means3D,
        scales,
        rotations,
        sh,
        opacities,
        raster_settings,
        cuda_args,
    ):

        # Restructure arguments the way that the C++ lib expects them
        args = (
            means3D,
            scales,
            rotations,
            sh,
            opacities,# 3dgs' parametes.
            raster_settings.scale_modifier,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            raster_settings.sh_degree,
            raster_settings.campos,
            raster_settings.prefiltered,
            raster_settings.debug,#raster_settings
            cuda_args
        )

        # TODO: update this. 
        num_rendered, means2D, depths, radii, cov3D, conic_opacity, rgb, clamped, tiles_touched = _C.preprocess_gaussians(*args)

        # Keep relevant tensors for backward
        ctx.raster_settings = raster_settings
        ctx.cuda_args = cuda_args
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(means3D, scales, rotations, sh, means2D, depths, radii, cov3D, conic_opacity, rgb, clamped, tiles_touched)
        ctx.mark_non_differentiable(radii, depths, tiles_touched)
        return means2D, rgb, conic_opacity, radii, depths, tiles_touched

    @staticmethod # TODO: gradient for conic_opacity is tricky. because cuda render backward generate dL_dconic and dL_dopacity sperately. 
    def backward(ctx, grad_means2D, grad_rgb, grad_conic_opacity, grad_radii, grad_depths, grad_tiles_touched):
        # grad_radii, grad_depths, grad_tiles_touched should be all None. 

        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        cuda_args = ctx.cuda_args
        means3D, scales, rotations, sh, means2D, depths, radii, cov3D, conic_opacity, rgb, clamped, tiles_touched = ctx.saved_tensors

        # Restructure args as C++ method expects them
        args = (radii,
                cov3D,
                clamped,#the above are all per-Gaussian intemediate results.
                means3D,
                scales,
                rotations, 
                sh, #input of this operator
                raster_settings.scale_modifier, 
                raster_settings.viewmatrix,
                raster_settings.projmatrix,
                raster_settings.tanfovx,
                raster_settings.tanfovy,
                raster_settings.image_height,
                raster_settings.image_width,
                raster_settings.sh_degree,
                raster_settings.campos,#rasterization setting.
                grad_means2D,
                grad_conic_opacity,
                grad_rgb,#gradients of output of this operator
                num_rendered,
                raster_settings.debug,
                cuda_args)

        dL_dmeans3D, dL_dscales, dL_drotations, dL_dsh, dL_dopacity = _C.preprocess_gaussians_backward(*args)

        grads = (
            dL_dmeans3D,
            dL_dscales,
            dL_drotations,
            dL_dsh,
            dL_dopacity,
            None,#raster_settings
            None,#raster_settings
        )

        return grads




########################### Render ###########################



def render_gaussians(
    means2D,
    conic_opacity,
    rgb,
    depths,
    radii,
    tiles_touched,
    raster_settings,
    cuda_args,
):
    return _RenderGaussians.apply(
        means2D,
        conic_opacity,
        rgb,
        depths,
        radii,
        tiles_touched,
        raster_settings,
        cuda_args,
    )

class _RenderGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means2D,
        conic_opacity,
        rgb,
        depths,
        radii,
        tiles_touched,
        raster_settings,
        cuda_args,
    ):

        # Restructure arguments the way that the C++ lib expects them
        args = (
            raster_settings.bg,
            raster_settings.image_height,
            raster_settings.image_width,# image setting
            means2D,
            depths,
            radii,
            conic_opacity,
            rgb,
            tiles_touched,# 3dgs intermediate results
            raster_settings.debug,
            cuda_args
        )

        num_rendered, color, n_render, n_consider, n_contrib, geomBuffer, binningBuffer, imgBuffer, distBuffer = _C.render_gaussians(*args)

        # Keep relevant tensors for backward
        ctx.raster_settings = raster_settings
        ctx.cuda_args = cuda_args
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(means2D, conic_opacity, rgb, geomBuffer, binningBuffer, imgBuffer, distBuffer)
        ctx.mark_non_differentiable(n_render, n_consider, n_contrib)
        return color, n_render, n_consider, n_contrib

    @staticmethod
    def backward(ctx, grad_color, grad_n_render, grad_n_consider, grad_n_contrib):
        # grad_n_render, grad_n_consider, grad_n_contrib should be all None. 

        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        cuda_args = ctx.cuda_args
        means2D, conic_opacity, rgb, geomBuffer, binningBuffer, imgBuffer, distBuffer = ctx.saved_tensors

        # Restructure args as C++ method expects them
        args = (raster_settings.bg,
                num_rendered,
                geomBuffer,
                binningBuffer,
                imgBuffer,
                distBuffer,# buffer
                grad_color,# gradient of output of this operator
                means2D,
                conic_opacity,
                rgb,# 3dgs intermediate results
                raster_settings.debug,
                cuda_args)

        dL_dmeans2D, dL_dconic_opacity, dL_dcolors = _C.render_gaussians_backward(*args)

        grads = (
            dL_dmeans2D,
            dL_dconic_opacity,
            dL_dcolors,
            None,
            None,
            None,
            None,
            None # this is for cuda_args
        )

        return grads





########################### Settings ###########################




class GaussianRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int 
    tanfovx : float
    tanfovy : float
    bg : torch.Tensor
    scale_modifier : float
    viewmatrix : torch.Tensor
    projmatrix : torch.Tensor
    sh_degree : int
    campos : torch.Tensor
    prefiltered : bool
    debug : bool

class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings

    def markVisible(self, positions):
        # Mark visible points (based on frustum culling for camera) with a boolean 
        with torch.no_grad():
            raster_settings = self.raster_settings
            visible = _C.mark_visible(
                positions,
                raster_settings.viewmatrix,
                raster_settings.projmatrix)
            
        return visible

    def forward(self, means3D, means2D, opacities, shs = None, colors_precomp = None, scales = None, rotations = None, cov3D_precomp = None, cuda_args = None):
        
        raster_settings = self.raster_settings

        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise Exception('Please provide excatly one of either SHs or precomputed colors!')
        
        if ((scales is None or rotations is None) and cov3D_precomp is None) or ((scales is not None or rotations is not None) and cov3D_precomp is not None):
            raise Exception('Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')
        
        if shs is None:
            shs = torch.Tensor([])
        if colors_precomp is None:
            colors_precomp = torch.Tensor([])

        if scales is None:
            scales = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])

        # Invoke C++/CUDA rasterization routine
        return rasterize_gaussians(
            means3D,
            means2D,
            shs,
            colors_precomp,
            opacities,
            scales, 
            rotations,
            cov3D_precomp,
            raster_settings, 
            cuda_args
        )

    def preprocess_gaussians(self, means3D, scales, rotations, shs, opacities, cuda_args = None):
        
        raster_settings = self.raster_settings

        # Invoke C++/CUDA rasterization routine
        return preprocess_gaussians(
            means3D,
            scales,
            rotations,
            shs,
            opacities,
            raster_settings,
            cuda_args)

    def render_gaussians(self, means2D, conic_opacity, rgb, depths, radii, tiles_touched, cuda_args = None):

        raster_settings = self.raster_settings

        # Invoke C++/CUDA rasterization routine
        return render_gaussians(
            means2D,
            conic_opacity,
            rgb,
            depths,
            radii,
            tiles_touched,
            raster_settings,
            cuda_args
        )
