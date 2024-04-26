import math
import time

import torch

from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
    GaussianRasterizerBatches,
)

num_gaussians = 50
num_batches=1
means3D = torch.randn(num_gaussians, 3).cuda()
scales = torch.randn(num_gaussians, 3).cuda()
rotations = torch.randn(num_gaussians, 3, 3).cuda()
shs = torch.randn(num_gaussians, 9).cuda()
opacity = torch.randn(num_gaussians, 1).cuda()

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
    batched_conic_opacity=[]
    batched_depths=[]
    batched_rgb=[]

    start_time = time.time()
    
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

        # TODO: make the below work
        # if mode == "train":
        #     means2D.retain_grad()

        batched_means2D.append(means2D)
        screenspace_params = [means2D, rgb, conic_opacity, radii, depths]
        batched_rasterizers.append(rasterizer)
        batched_screenspace_params.append(screenspace_params)
        batched_radii.append(radii)
        batched_rgb.append(rgb)
        batched_conic_opacity.append(conic_opacity)
        batched_depths.append(depths)


    end_time = time.time()
    preprocess_time = end_time - start_time
    print(f"Time taken by test_batched_gaussian_rasterizer: {preprocess_time:.4f} seconds")
    # Perform further operations with the batched results
    # Test results and performance
   
    batched_means2D = torch.stack(batched_means2D, dim=0)
    batched_radii = torch.stack(batched_radii, dim=0)
    batched_conic_opacity=torch.stack(batched_conic_opacity,dim=0)
    batched_rgb=torch.stack(batched_rgb,dim=0)
    batched_depths=torch.stack(batched_depths,dim=0)
    
    return batched_means2D, batched_radii, batched_screenspace_params,batched_conic_opacity,batched_rgb,batched_depths
    
    
def test_batched_gaussian_rasterizer_batch_processing():
    # Set up the input data
    start_time = time.time()
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
    batched_tanfovx = torch.tensor([math.tan(camera.FoVx * 0.5) for camera in batched_viewpoint_cameras]).cuda()
    batched_tanfovy = torch.tensor([math.tan(camera.FoVy * 0.5) for camera in batched_viewpoint_cameras]).cuda()
    batched_viewmatrix = torch.stack([camera.world_view_transform for camera in batched_viewpoint_cameras]).cuda()
    batched_projmatrix = torch.stack([camera.full_proj_transform for camera in batched_viewpoint_cameras]).cuda()
    batched_campos = torch.stack([camera.camera_center for camera in batched_viewpoint_cameras]).cuda()
    
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
    rasterizer = GaussianRasterizerBatches(raster_settings=batched_raster_settings)

    # Set up CUDA arguments for the batch
    cuda_args = get_cuda_args(batched_strategies[0], mode)  # TODO: Check if this is correct for the batch

    # Preprocess the Gaussians for the entire batch
    batched_means2D, batched_rgb, batched_conic_opacity, batched_radii, batched_depths = rasterizer.preprocess_gaussians(
        means3D=means3D,
        scales=scales,
        rotations=rotations,
        shs=shs,
        opacities=opacity,
        batched_cuda_args=cuda_args
    )
    end_time = time.time()
    preprocess_time = end_time - start_time
    print(f"Time taken by test_batched_gaussian_rasterizer_batch_processing: {preprocess_time:.4f} seconds")

    # TODO: make the below work
    # if mode == "train":
    #     batched_means2D.retain_grad()


    # Perform assertions on the preprocessed data
    
    assert batched_means2D.shape == (num_batches, num_gaussians, 2)
    assert batched_rgb.shape == (num_batches, num_gaussians, 3)
    assert batched_conic_opacity.shape == (num_batches, num_gaussians,4)
    assert batched_radii.shape == (num_batches, num_gaussians)
    assert batched_depths.shape == (num_batches, num_gaussians)
    torch.cuda.empty_cache()
    
    batched_screenspace_params = []
    for i in range(num_batches):
        means2D = batched_means2D[i]
        rgb = batched_rgb[i]
        conic_opacity = batched_conic_opacity[i]
        radii = batched_radii[i]
        depths = batched_depths[i]
        
        screenspace_params = [means2D, rgb, conic_opacity, radii, depths]
        batched_screenspace_params.append(screenspace_params)
    
    return batched_means2D, batched_radii, batched_screenspace_params, batched_conic_opacity,batched_rgb,batched_depths


def compare_tensors(tensor1, tensor2):
    if tensor1.shape != tensor2.shape:
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

if __name__ == "__main__":
    batched_means2D, batched_radii, batched_screenspace_params,batched_conic_opacity,batched_rgb,batched_depths = test_batched_gaussian_rasterizer()
    batched_means2D_batch_processed, batched_radii_batch_processed, batched_screenspace_params_batch_processed,batched_conic_opacity_batch_processed,batched_rgb_batch_processed,batched_depths_batch_processed = test_batched_gaussian_rasterizer_batch_processing()
        
    assert compare_tensors(batched_means2D, batched_means2D_batch_processed)
    assert compare_tensors(batched_radii, batched_radii_batch_processed)
    assert compare_tensors(batched_conic_opacity, batched_conic_opacity_batch_processed)
    print(batched_rgb)
    print('*****')
    print(batched_rgb_batch_processed)

    assert compare_tensors(batched_rgb, batched_rgb_batch_processed)
    assert compare_tensors(batched_depths, batched_depths_batch_processed)
    assert len(batched_screenspace_params) == len(batched_screenspace_params_batch_processed)
    
