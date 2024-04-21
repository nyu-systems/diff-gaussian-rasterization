import math
import time

import torch

from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)


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

def test_gaussian_rasterizer_time():
    # Set up the input data
    num_gaussians = 10000
    means3D = torch.randn(num_gaussians, 3).cuda()
    scales = torch.randn(num_gaussians, 3).cuda()
    rotations = torch.randn(num_gaussians, 3, 3).cuda()
    shs = torch.randn(num_gaussians, 9).cuda()
    opacities = torch.randn(num_gaussians, 1).cuda()

    # Set up the rasterization settings
    image_height = 512
    image_width = 512
    tanfovx = 1.0
    tanfovy = 1.0
    bg = torch.ones(3).cuda()
    scale_modifier = 1.0
    viewmatrix = torch.eye(4).cuda()
    projmatrix = torch.eye(4).cuda()
    sh_degree = 2
    campos = torch.zeros(3).cuda()
    prefiltered = False
    debug = False
    
    # mode="train"
    # strategy=None
    # cuda_args = get_cuda_args(strategy, mode)

    raster_settings = GaussianRasterizationSettings(
        image_height, image_width, tanfovx, tanfovy, bg,
        scale_modifier, viewmatrix, projmatrix, sh_degree,
        campos, prefiltered, debug
    )

    # Create the GaussianRasterizer
    rasterizer = GaussianRasterizer(raster_settings)

    # Measure the time for preprocess_gaussians
    start_time = time.time()
    means2D, rgb, conic_opacity, radii, depths = rasterizer.preprocess_gaussians(
        means3D, scales, rotations, shs, opacities
    )
    end_time = time.time()

    preprocess_time = end_time - start_time
    print(f"Time taken by preprocess_gaussians: {preprocess_time:.4f} seconds")


def test_batched_gaussian_rasterizer():
    # Set up the input data
    num_gaussians = 10000
    num_batches = 4
    means3D = torch.randn(num_gaussians, 3).cuda()
    scales = torch.randn(num_gaussians, 3).cuda()
    rotations = torch.randn(num_gaussians, 3, 3).cuda()
    shs = torch.randn(num_gaussians, 9).cuda()
    opacity = torch.randn(num_gaussians, 1).cuda()

    # Set up the viewpoint cameras
    batched_viewpoint_cameras = []
    for _ in range(num_batches):
        viewpoint_camera = type('ViewpointCamera', (), {})
        viewpoint_camera.FoVx = math.radians(60)
        viewpoint_camera.FoVy = math.radians(60)
        viewpoint_camera.image_height = 512
        viewpoint_camera.image_width = 512
        viewpoint_camera.world_view_transform = torch.eye(4).cuda()
        viewpoint_camera.full_proj_transform = torch.eye(4).cuda()
        viewpoint_camera.camera_center = torch.zeros(3).cuda()
        batched_viewpoint_cameras.append(viewpoint_camera)

    # Set up the strategies
    batched_strategies = [None] * num_batches

    # Set up other parameters
    bg_color = torch.ones(3).cuda()
    scaling_modifier = 1.0
    pc = type('PC', (), {})
    pc.active_sh_degree = 2
    pipe = type('Pipe', (), {})
    pipe.debug = False
    mode = "train"

    batched_rasterizers = []
    batched_cuda_args = []
    batched_screenspace_params = []
    batched_means2D = []
    batched_radii = []

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
            debug=pipe.debug
        )
        rasterizer = GaussianRasterizer(raster_settings=raster_settings)
        ########## [END] Prepare CUDA Rasterization Settings ##########

        #[3DGS-wise preprocess]
        means2D, rgb, conic_opacity, radii, depths = rasterizer.preprocess_gaussians(
            means3D=means3D,
            scales=scales,
            rotations=rotations,
            shs=shs,
            opacities=opacity,
            cuda_args=cuda_args
        )

        if mode == "train":
            means2D.retain_grad()

        batched_means2D.append(means2D)
        screenspace_params = [means2D, rgb, conic_opacity, radii, depths]
        batched_rasterizers.append(rasterizer)
        batched_screenspace_params.append(screenspace_params)
        batched_radii.append(radii)

    # Perform further operations with the batched results
    # Test results and performance
    
def test_batched_gaussian_rasterizer_batch_processing():
    # Set up the input data
    num_gaussians = 10000
    num_batches = 4
    means3D = torch.randn(num_gaussians, 3).cuda()
    scales = torch.randn(num_gaussians, 3).cuda()
    rotations = torch.randn(num_gaussians, 3, 3).cuda()
    shs = torch.randn(num_gaussians, 9).cuda()
    opacity = torch.randn(num_gaussians, 1).cuda()

    # Set up the viewpoint cameras
    batched_viewpoint_cameras = []
    for _ in range(num_batches):
        viewpoint_camera = type('ViewpointCamera', (), {})
        viewpoint_camera.FoVx = math.radians(60)
        viewpoint_camera.FoVy = math.radians(60)
        viewpoint_camera.image_height = 512
        viewpoint_camera.image_width = 512
        viewpoint_camera.world_view_transform = torch.eye(4).cuda()
        viewpoint_camera.full_proj_transform = torch.eye(4).cuda()
        viewpoint_camera.camera_center = torch.zeros(3).cuda()
        batched_viewpoint_cameras.append(viewpoint_camera)

    # Set up the strategies
    batched_strategies = [None] * num_batches

    # Set up other parameters
    bg_color = torch.ones(3).cuda()
    scaling_modifier = 1.0
    pc = type('PC', (), {})
    pc.active_sh_degree = 2
    pipe = type('Pipe', (), {})
    pipe.debug = False
    mode = "train"

    # Set up rasterization configuration for the batch
    batched_tanfovx = [math.tan(camera.FoVx * 0.5) for camera in batched_viewpoint_cameras]
    batched_tanfovy = [math.tan(camera.FoVy * 0.5) for camera in batched_viewpoint_cameras]
    batched_viewmatrix = [camera.world_view_transform for camera in batched_viewpoint_cameras]
    batched_projmatrix = [camera.full_proj_transform for camera in batched_viewpoint_cameras]
    batched_campos = [camera.camera_center for camera in batched_viewpoint_cameras]

    batched_raster_settings = GaussianRasterizationSettings(
        image_height=int(batched_viewpoint_cameras[0].image_height),
        image_width=int(batched_viewpoint_cameras[0].image_width),
        tanfovx=batched_tanfovx,
        tanfovy=batched_tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=batched_viewmatrix,
        projmatrix=batched_projmatrix,
        sh_degree=pc.active_sh_degree,
        campos=batched_campos,
        prefiltered=False,
        debug=pipe.debug
    )

    # Create the GaussianRasterizer for the batch
    rasterizer = GaussianRasterizer(raster_settings=batched_raster_settings)

    # Set up CUDA arguments for the batch
    cuda_args = get_cuda_args(batched_strategies[0], mode)  # TODO: Check if this is correct for the batch

    # Preprocess the Gaussians for the entire batch
    batched_means2D, batched_rgb, batched_conic_opacity, batched_radii, batched_depths = rasterizer.preprocess_gaussians(
        means3D=means3D,
        scales=scales,
        rotations=rotations,
        shs=shs,
        opacities=opacity,
        cuda_args=cuda_args
    )

    if mode == "train":
        batched_means2D.retain_grad()

    batched_screenspace_params = [batched_means2D, batched_rgb, batched_conic_opacity, batched_radii, batched_depths]

    # Perform assertions on the preprocessed data
    assert batched_means2D.shape == (num_gaussians, num_batches, 2)
    assert batched_rgb.shape == (num_gaussians, num_batches, 3)
    assert batched_conic_opacity.shape == (num_gaussians, num_batches, 1)
    assert batched_radii.shape == (num_gaussians, num_batches)
    assert batched_depths.shape == (num_gaussians, num_batches)


if __name__ == "__main__":
    test_gaussian_rasterizer_time()
    test_batched_gaussian_rasterizer_batch_processing()
    