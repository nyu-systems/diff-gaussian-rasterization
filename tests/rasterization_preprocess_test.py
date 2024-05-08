import math

import pytest
import torch

from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
    GaussianRasterizerBatches,
)

num_gaussians = 1000000
num_batches = 64
SH_ACTIVE_DEGREE = 3


@pytest.fixture(scope="module")
def setup_data():
    # Set up the input data, viewpoint cameras, strategies, etc.
    means3D = torch.randn(num_gaussians, 3).cuda()
    scales = torch.randn(num_gaussians, 3).cuda()
    rotations = torch.randn(num_gaussians, 4).cuda()
    shs = torch.randn(num_gaussians, 16, 3).cuda()
    opacity = torch.randn(num_gaussians, 1).cuda()

    means3D.requires_grad = True
    scales.requires_grad = True
    rotations.requires_grad = True
    shs.requires_grad = True
    opacity.requires_grad = True

    batched_viewpoint_cameras = []
    for _ in range(num_batches):
        viewpoint_camera = type("ViewpointCamera", (), {})
        viewpoint_camera.FoVx = math.radians(60)
        viewpoint_camera.FoVy = math.radians(60)
        viewpoint_camera.image_height = 512
        viewpoint_camera.image_width = 512
        viewpoint_camera.world_view_transform = torch.eye(4).cuda()
        viewpoint_camera.full_proj_transform = torch.eye(4).cuda()
        viewpoint_camera.camera_center = torch.zeros(3).cuda()
        batched_viewpoint_cameras.append(viewpoint_camera)

    batched_strategies = [None] * num_batches

    bg_color = torch.ones(3).cuda()
    scaling_modifier = 1.0
    pc = type("PC", (), {})
    pc.active_sh_degree = SH_ACTIVE_DEGREE
    pipe = type("Pipe", (), {})
    pipe.debug = False
    mode = "train"

    return (
        means3D,
        scales,
        rotations,
        shs,
        opacity,
        batched_viewpoint_cameras,
        batched_strategies,
        bg_color,
        scaling_modifier,
        pc,
        pipe,
        mode,
    )


def compute_dummy_loss(means3D, scales, rotations, shs, opacity):
    losses = [(tensor - torch.ones_like(tensor)).pow(2).mean() for tensor in [means3D, scales, rotations, shs, opacity]]
    loss = sum(losses)
    return loss


def zero_grad(means3D, scales, rotations, shs, opacity):
    means3D.grad = None
    scales.grad = None
    rotations.grad = None
    shs.grad = None
    opacity.grad = None


def get_cuda_args(strategy, mode="train"):
    cuda_args = {
        "mode": mode,
        "world_size": "1",
        "global_rank": "0",
        "local_rank": "0",
        "mp_world_size": "1",
        "mp_rank": "0",
        "log_folder": "./logs",
        "log_interval": "10",
        "iteration": "0",
        "zhx_debug": "False",
        "zhx_time": "False",
        "dist_global_strategy": "default",
        "avoid_pixel_all2all": False,
        "stats_collector": {},
    }
    return cuda_args


def run_batched_gaussian_rasterizer(setup_data):
    (
        means3D,
        scales,
        rotations,
        shs,
        opacity,
        batched_viewpoint_cameras,
        batched_strategies,
        bg_color,
        scaling_modifier,
        pc,
        pipe,
        mode,
    ) = setup_data

    batched_rasterizers = []
    batched_cuda_args = []
    batched_screenspace_params = []
    batched_means2D = []
    batched_radii = []
    batched_conic_opacity = []
    batched_depths = []
    batched_rgb = []

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start_event.record()
    for i, (viewpoint_camera, strategy) in enumerate(zip(batched_viewpoint_cameras, batched_strategies)):
        ########## [START] Prepare CUDA Rasterization Settings ##########
        cuda_args = get_cuda_args(strategy, mode)
        batched_cuda_args.append(cuda_args)

        # Set up rasterization configuration
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=pipe.debug,
        )
        rasterizer = GaussianRasterizer(raster_settings=raster_settings)
        ########## [END] Prepare CUDA Rasterization Settings ##########

        # [3DGS-wise preprocess]
        means2D, rgb, conic_opacity, radii, depths = rasterizer.preprocess_gaussians(
            means3D=means3D, scales=scales, rotations=rotations, shs=shs, opacities=opacity, cuda_args=cuda_args
        )

        batched_means2D.append(means2D)
        screenspace_params = [means2D, rgb, conic_opacity, radii, depths]
        batched_rasterizers.append(rasterizer)
        batched_screenspace_params.append(screenspace_params)
        batched_radii.append(radii)
        batched_rgb.append(rgb)
        batched_conic_opacity.append(conic_opacity)
        batched_depths.append(depths)

    end_event.record()
    torch.cuda.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event)
    print(f"Time taken by test_batched_gaussian_rasterizer: {elapsed_time_ms:.4f} ms")

    batched_means2D = torch.stack(batched_means2D, dim=0)
    batched_radii = torch.stack(batched_radii, dim=0)
    batched_conic_opacity = torch.stack(batched_conic_opacity, dim=0)
    batched_rgb = torch.stack(batched_rgb, dim=0)
    batched_depths = torch.stack(batched_depths, dim=0)

    zero_grad(means3D, scales, rotations, shs, opacity)
    start_backward_event = torch.cuda.Event(enable_timing=True)
    end_backward_event = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start_backward_event.record()
    
    loss = compute_dummy_loss(means3D, scales, rotations, shs, opacity)
    loss.backward()
    
    end_backward_event.record()
    torch.cuda.synchronize()
    backward_time_ms = start_backward_event.elapsed_time(end_backward_event)
    print(f"Time taken by run_batched_gaussian_rasterizer BACKWARD: {backward_time_ms:.4f} ms")

    assert means3D.grad is not None, "Means3D gradient is None."
    assert scales.grad is not None, "Scales gradient is None."
    assert rotations.grad is not None, "Rotations gradient is None."
    assert shs.grad is not None, "SHs gradient is None."
    assert opacity.grad is not None, "Opacity gradient is None."

    return (
        batched_means2D,
        batched_radii,
        batched_screenspace_params,
        batched_conic_opacity,
        batched_rgb,
        batched_depths,
        means3D.grad.clone(),
        scales.grad.clone(),
        rotations.grad.clone(),
        shs.grad.clone(),
        opacity.grad.clone(),
    )


def run_batched_gaussian_rasterizer_batch_processing(setup_data):
    (
        means3D,
        scales,
        rotations,
        shs,
        opacity,
        batched_viewpoint_cameras,
        batched_strategies,
        bg_color,
        scaling_modifier,
        pc,
        pipe,
        mode,
    ) = setup_data

    # Set up the input data
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start_event.record()

    # Set up rasterization configuration for the batch
    raster_settings_batch = []
    batched_cuda_args = []
    for i, (viewpoint_camera, strategy) in enumerate(zip(batched_viewpoint_cameras, batched_strategies)):
        ########## [START] Prepare CUDA Rasterization Settings ##########
        cuda_args = get_cuda_args(strategy, mode)
        batched_cuda_args.append(cuda_args)
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

        raster_settings = GaussianRasterizationSettings(
            image_height=int(batched_viewpoint_cameras[0].image_height),
            image_width=int(batched_viewpoint_cameras[0].image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=pipe.debug,
        )
        raster_settings_batch.append(raster_settings)

    # Create the GaussianRasterizer for the batch
    rasterizer = GaussianRasterizerBatches(raster_settings_batch=raster_settings_batch)

    # Preprocess the Gaussians for the entire batch
    (
        batched_means2D,
        batched_rgb,
        batched_conic_opacity,
        batched_radii,
        batched_depths,
    ) = rasterizer.preprocess_gaussians(
        means3D=means3D,
        scales=scales,
        rotations=rotations,
        shs=shs,
        opacities=opacity,
        batched_cuda_args=batched_cuda_args[0],  # TODO: look into sending list of cuda_args/strategies
    )
    end_event.record()
    torch.cuda.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event)
    print(f"Time taken by test_batched_gaussian_rasterizer: {elapsed_time_ms:.4f} ms")

    # Perform assertions on the preprocessed data

    assert batched_means2D.shape == (num_batches, num_gaussians, 2)
    assert batched_rgb.shape == (num_batches, num_gaussians, 3)
    assert batched_conic_opacity.shape == (num_batches, num_gaussians, 4)
    assert batched_radii.shape == (num_batches, num_gaussians)
    assert batched_depths.shape == (num_batches, num_gaussians)

    batched_screenspace_params = []
    for i in range(num_batches):
        means2D = batched_means2D[i]
        rgb = batched_rgb[i]
        conic_opacity = batched_conic_opacity[i]
        radii = batched_radii[i]
        depths = batched_depths[i]

        screenspace_params = [means2D, rgb, conic_opacity, radii, depths]
        batched_screenspace_params.append(screenspace_params)

    zero_grad(means3D, scales, rotations, shs, opacity)
    
    start_backward_event = torch.cuda.Event(enable_timing=True)
    end_backward_event = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start_backward_event.record()

    loss = compute_dummy_loss(means3D, scales, rotations, shs, opacity)
    loss.backward()

    end_backward_event.record()
    torch.cuda.synchronize()
    backward_time_ms = start_backward_event.elapsed_time(end_backward_event)
    print(f"Time taken by run_batched_gaussian_rasterizer_batch_processing BACKWARD: {backward_time_ms:.4f} ms")

    assert means3D.grad is not None, "Means3D gradient is None."
    assert scales.grad is not None, "Scales gradient is None."
    assert rotations.grad is not None, "Rotations gradient is None."
    assert shs.grad is not None, "SHs gradient is None."
    assert opacity.grad is not None, "Opacity gradient is None."

    return (
        batched_means2D,
        batched_radii,
        batched_screenspace_params,
        batched_conic_opacity,
        batched_rgb,
        batched_depths,
        means3D.grad.clone(),
        scales.grad.clone(),
        rotations.grad.clone(),
        shs.grad.clone(),
        opacity.grad.clone(),
    )


def compare_tensors(tensor1, tensor2):
    if tensor1 is None and tensor2 is None:
        return True
    elif tensor1 is None or tensor2 is None:
        print("One of the tensors is None.")
        return False
    elif tensor1.shape != tensor2.shape:
        print("Tensors have different shapes:")
        print("Tensor 1 shape:", tensor1.shape)
        print("Tensor 2 shape:", tensor2.shape)
        return False

    equality_matrix = torch.eq(tensor1, tensor2)
    if torch.all(equality_matrix):
        return True
    else:
        print("Tensors have non-matching values.")
        non_matching_indices = torch.where(equality_matrix == False)
        for idx in zip(*non_matching_indices[:5]):
            value1 = tensor1[idx].item()
            value2 = tensor2[idx].item()
            print(f"Non-matching values at index {idx}: {value1} != {value2}")
        return False


def test_compare_batched_gaussian_rasterizer_results(setup_data):
    (
        batched_means2D,
        batched_radii,
        batched_screenspace_params,
        batched_conic_opacity,
        batched_rgb,
        batched_depths,
        batched_dL_means3D,
        batched_dL_scales,
        batched_dL_rotations,
        batched_dL_shs,
        batched_dL_opacity,
    ) = run_batched_gaussian_rasterizer(setup_data)
    (
        batched_means2D_batch_processed,
        batched_radii_batch_processed,
        batched_screenspace_params_batch_processed,
        batched_conic_opacity_batch_processed,
        batched_rgb_batch_processed,
        batched_depths_batch_processed,
        batched_dL_means3D_batch_processed,
        batched_dL_scales_batch_processed,
        batched_dL_rotations_batch_processed,
        batched_dL_shs_batch_processed,
        batched_dL_opacity_batch_processed,
    ) = run_batched_gaussian_rasterizer_batch_processing(setup_data)

    assert compare_tensors(batched_means2D, batched_means2D_batch_processed), "Means2D do not match."
    assert compare_tensors(batched_radii, batched_radii_batch_processed), "Radii do not match."
    assert compare_tensors(batched_conic_opacity, batched_conic_opacity_batch_processed), "Conic opacity do not match."

    assert compare_tensors(batched_rgb, batched_rgb_batch_processed), "RGB values do not match."
    assert compare_tensors(batched_depths, batched_depths_batch_processed), "Depths do not match."
    assert len(batched_screenspace_params) == len(
        batched_screenspace_params_batch_processed
    ), "Screenspace params do not match."

    # -------BACKWARD PASS-------
    assert compare_tensors(batched_dL_means3D, batched_dL_means3D_batch_processed), "dL_means3D do not match."
    assert compare_tensors(batched_dL_scales, batched_dL_scales_batch_processed), "dL_scales do not match."
    assert compare_tensors(batched_dL_rotations, batched_dL_rotations_batch_processed), "dL_rotations do not match."
    assert compare_tensors(batched_dL_shs, batched_dL_shs_batch_processed), "dL_shs do not match."
    assert compare_tensors(batched_dL_opacity, batched_dL_opacity_batch_processed), "dL_opacity do not match."
