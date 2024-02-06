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

#include "rasterizer_impl.h"
#include "timers.cu"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <string>
#include <cstdlib>
#include <chrono>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

#include "auxiliary.h"
#include "forward.h"
#include "backward.h"

// Helper function to find the next-highest bit of the MSB
// on the CPU.
uint32_t getHigherMsb(uint32_t n)
{
	uint32_t msb = sizeof(n) * 4;
	uint32_t step = msb;
	while (step > 1)
	{
		step /= 2;
		if (n >> msb)
			msb += step;
		else
			msb -= step;
	}
	if (n >> msb)
		msb++;
	return msb;
}

// Wrapper method to call auxiliary coarse frustum containment test.
// Mark all Gaussians that pass it.
__global__ void checkFrustum(int P,
	const float* orig_points,
	const float* viewmatrix,
	const float* projmatrix,
	bool* present)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	float3 p_view;
	present[idx] = in_frustum(idx, orig_points, viewmatrix, projmatrix, false, p_view);
}

// Generates one key/value pair for all Gaussian / tile overlaps. 
// Run once per Gaussian (1:N mapping).
__global__ void duplicateWithKeys(
	int P,
	const float2* points_xy,
	const float* depths,
	const uint32_t* offsets,
	uint64_t* gaussian_keys_unsorted,
	uint32_t* gaussian_values_unsorted,
	int* radii,
	bool* compute_locally,
	dim3 grid,
	int local_rank,
	int world_size)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Generate no key/value pair for invisible Gaussians
	if (radii[idx] > 0)
	{
		// Find this Gaussian's offset in buffer for writing keys/values.
		uint32_t off = (idx == 0) ? 0 : offsets[idx - 1];
		uint2 rect_min, rect_max;

		getRect(points_xy[idx], radii[idx], rect_min, rect_max, grid);

		// For each tile that the bounding rect overlaps, emit a 
		// key/value pair. The key is |  tile ID  |      depth      |,
		// and the value is the ID of the Gaussian. Sorting the values 
		// with this key yields Gaussian IDs in a list, such that they
		// are first sorted by tile and then by depth. 
		for (int y = rect_min.y; y < rect_max.y; y++)
		{//TODO: this has a small problem when rect_min.y == rect_max.y; this is also a reasonable case.
			for (int x = rect_min.x; x < rect_max.x; x++)
			if (compute_locally[y * grid.x + x])
			{
				uint64_t key = y * grid.x + x;
				key <<= 32;
				key |= *((uint32_t*)&depths[idx]);
				gaussian_keys_unsorted[off] = key;
				gaussian_values_unsorted[off] = idx;
				off++;
			}
		}
	}
}

// Check keys to see if it is at the start/end of one tile's range in 
// the full sorted list. If yes, write start/end of this tile. 
// Run once per instanced (duplicated) Gaussian ID.
__global__ void identifyTileRanges(int L, uint64_t* point_list_keys, uint2* ranges)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= L)
		return;

	// Read tile ID from key. Update start/end of tile range if at limit.
	uint64_t key = point_list_keys[idx];
	uint32_t currtile = key >> 32;
	if (idx == 0)
		ranges[currtile].x = 0;
	else
	{
		uint32_t prevtile = point_list_keys[idx - 1] >> 32;
		if (currtile != prevtile)
		{
			ranges[prevtile].y = idx;
			ranges[currtile].x = idx;
		}
	}
	if (idx == L - 1)
		ranges[currtile].y = L;
}

// Mark Gaussians as visible/invisible, based on view frustum testing
void CudaRasterizer::Rasterizer::markVisible(
	int P,
	float* means3D,
	float* viewmatrix,
	float* projmatrix,
	bool* present)
{
	checkFrustum << <(P + 255) / 256, 256 >> > (
		P,
		means3D,
		viewmatrix, projmatrix,
		present);
}

CudaRasterizer::GeometryState CudaRasterizer::GeometryState::fromChunk(char*& chunk, size_t P, bool sep_rendering=false)
{
	GeometryState geom;
	if (!sep_rendering)
	{
		obtain(chunk, geom.depths, P, 128);
		obtain(chunk, geom.clamped, P * 3, 128);
		obtain(chunk, geom.internal_radii, P, 128);
		obtain(chunk, geom.means2D, P, 128);
		obtain(chunk, geom.cov3D, P * 6, 128);
		obtain(chunk, geom.conic_opacity, P, 128);
		obtain(chunk, geom.rgb, P * 3, 128);
	}
	obtain(chunk, geom.tiles_touched, P, 128);
	cub::DeviceScan::InclusiveSum(nullptr, geom.scan_size, geom.tiles_touched, geom.tiles_touched, P);
	// no work is done and the required allocation size is returned in geom.scan_size.
	obtain(chunk, geom.scanning_space, geom.scan_size, 128);
	obtain(chunk, geom.point_offsets, P, 128);
	return geom;
}

CudaRasterizer::ImageState CudaRasterizer::ImageState::fromChunk(char*& chunk, size_t N)
{
	ImageState img;
	obtain(chunk, img.accum_alpha, N, 128);
	obtain(chunk, img.n_contrib, N, 128);
	obtain(chunk, img.n_contrib2loss, N, 128);
	obtain(chunk, img.ranges, N, 128);
	return img;
}

CudaRasterizer::BinningState CudaRasterizer::BinningState::fromChunk(char*& chunk, size_t P)
{
	BinningState binning;
	obtain(chunk, binning.point_list, P, 128);
	obtain(chunk, binning.point_list_unsorted, P, 128);
	obtain(chunk, binning.point_list_keys, P, 128);
	obtain(chunk, binning.point_list_keys_unsorted, P, 128);
	cub::DeviceRadixSort::SortPairs(
		nullptr, binning.sorting_size,
		binning.point_list_keys_unsorted, binning.point_list_keys,
		binning.point_list_unsorted, binning.point_list, P);
	obtain(chunk, binning.list_sorting_space, binning.sorting_size, 128);
	return binning;
}

CudaRasterizer::DistributedState CudaRasterizer::DistributedState::fromChunk(char*& chunk, size_t tile_num, bool sep_rendering=false)
{
	DistributedState dist;
	obtain(chunk, dist.gs_on_tiles, tile_num, 128);
	obtain(chunk, dist.gs_on_tiles_offsets, tile_num, 128);
	cub::DeviceScan::InclusiveSum(nullptr, dist.scan_size, dist.gs_on_tiles, dist.gs_on_tiles_offsets, tile_num);
	obtain(chunk, dist.scanning_space, dist.scan_size, 128);
	if (!sep_rendering)
		obtain(chunk, dist.compute_locally, tile_num, 128);
	return dist;
}

__global__ void get_n_render(
	const int tile_num,
	const uint2* ranges,
	int* n_render
) {
	auto idx = cg::this_grid().thread_rank();
	if (idx >= tile_num)
		return;
	n_render[idx] = ranges[idx].y - ranges[idx].x;
}

__global__ void reduce_data_per_block(
	const int width,
	const int height,
	const uint32_t* n_data_per_pixel,
	int* n_data_per_block,
	bool* compute_locally
) {
	auto block = cg::this_thread_block();
	if (!compute_locally[block.group_index().y * gridDim.x + block.group_index().x])
		return;
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	int pix_id = width * pix.y + pix.x;

	int blocksz = block.size(), beta;
	__shared__ int reduction_s[BLOCK_X * BLOCK_Y];
	int tid = block.thread_rank();

	bool inside = pix.x < width && pix.y < height;
	int data = inside ? (int)n_data_per_pixel[pix_id] : 0;// TODO: make sure the explicit cast is correct, i.e. n_data_per_pixel < 2^31-1;

	cg::thread_block_tile<32> tile = cg::tiled_partition<32>(block);
	reduction_s[tid] = cg::reduce(tile, data, cg::plus<int>());
    cg::sync(block);

    if (tid == 0) {
        beta = 0;
        for (int i = 0; i < blocksz; i += tile.num_threads()) {
            beta += reduction_s[i];
        }
        n_data_per_block[block.group_index().y * gridDim.x + block.group_index().x] = beta;
    }
}

__global__ void getComputeLocally(//TODO: this function is not heavy enough to be parallelized.
	const int tile_num,
	uint32_t* gs_on_tiles_offsets,
	bool* compute_locally,
	int last_local_num_rendered_end,
	int local_num_rendered_end
) {
	auto idx = cg::this_grid().thread_rank();
	if (idx >= tile_num)
		return;

	int x = (int)gs_on_tiles_offsets[idx];
	if (x > last_local_num_rendered_end && x <= local_num_rendered_end)
		compute_locally[idx] = true;
	else
		compute_locally[idx] = false;
}

__global__ void getComputeLocallyByTileNum(//TODO: this function is not heavy enough to be parallelized.
	const int tile_num,
	bool* compute_locally,
	int last_local_num_rendered_end,
	int local_num_rendered_end
) {
	auto idx = cg::this_grid().thread_rank();
	if (idx >= tile_num)
		return;

	if (idx >= last_local_num_rendered_end && idx < local_num_rendered_end)
		compute_locally[idx] = true;
	else
		compute_locally[idx] = false;
}

__global__ void getComputeLocallyByTileId(
	const int tile_num,
	bool* compute_locally,
	int tile_id_start,
	int tile_id_end
) {
	auto idx = cg::this_grid().thread_rank();
	if (idx >= tile_num)
		return;

	if (idx >= tile_id_start && idx < tile_id_end)
		compute_locally[idx] = true;
	else
		compute_locally[idx] = false;
}

__global__ void getComputeLocallyByRowId(
	const int tile_num,
	bool* compute_locally,
	int tile_grid_x,
	int tile_grid_y,
	int row_id_start,
	int row_id_end
) {
	auto idx = cg::this_grid().thread_rank();
	if (idx >= tile_num)
		return;

	int tile_x = idx % tile_grid_x;
	int tile_y = idx / tile_grid_x;
	if (tile_y >= row_id_start && tile_y < row_id_end)
		compute_locally[idx] = true;
	else
		compute_locally[idx] = false;
}


__global__ void updateTileTouched(
	const int P,
	const dim3 tile_grid,
	int* radii,
	float2* means2D,
	uint32_t* tiles_touched,
	bool* compute_locally
) {
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	uint32_t cnt = 0;
	if (radii[idx] > 0)
	{
		uint2 rect_min, rect_max;
		getRect(means2D[idx], radii[idx], rect_min, rect_max, tile_grid);
		for (int y = rect_min.y; y < rect_max.y; y++)
			for (int x = rect_min.x; x < rect_max.x; x++)
			if (compute_locally[y * tile_grid.x + x])
				cnt++;
	}

	tiles_touched[idx] = cnt;
}

__global__ void getGlobalGaussianOnTiles(//TODO: maybe this could take significant amount of time. 
	const int P,
	const float2* means2D,
	int* radii,
	const dim3 tile_grid,
	uint32_t* gs_on_tiles
) {
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	if (radii[idx] > 0)
	{
		uint2 rect_min, rect_max;
		getRect(means2D[idx], radii[idx], rect_min, rect_max, tile_grid);
		for (int y = rect_min.y; y < rect_max.y; y++)
			for (int x = rect_min.x; x < rect_max.x; x++)
			{
				atomicAdd(&gs_on_tiles[y * tile_grid.x + x], 1);
				//TODO: Do I have to use atomicAdd? This is slow, honestly. 
			}
	}
}

// NOTE: This method should also deal with world_size == 1 safely.
void updateDistributedStatLocally(//TODO: optimize implementations for all these kernels. 
	const int P,
	const int width,
	const int height,
	const dim3 tile_grid,
	int* radii,
	float2* means2D,
	CudaRasterizer::DistributedState& distState,
	const int local_rank,
	const int world_size,
	const char * dist_division_mode,
	MyTimerOnGPU& timer
){
	int tile_num = tile_grid.x * tile_grid.y;
	timer.start("21 updateDistributedStatLocally.getGlobalGaussianOnTiles");
	cudaMemset(distState.gs_on_tiles, 0, tile_num * sizeof(uint32_t));
	getGlobalGaussianOnTiles <<<(P + 255) / 256, 256 >>> (
		P,
		means2D,
		radii,
		tile_grid,
		distState.gs_on_tiles
	);
	timer.stop("21 updateDistributedStatLocally.getGlobalGaussianOnTiles");

	// getComputeLocally
	if (world_size >= 1) {
		timer.start("22 updateDistributedStatLocally.InclusiveSum");
		cub::DeviceScan::InclusiveSum(distState.scanning_space, distState.scan_size, distState.gs_on_tiles, distState.gs_on_tiles_offsets, tile_num);
		timer.stop("22 updateDistributedStatLocally.InclusiveSum");

		int num_rendered;
		cudaMemcpy(&num_rendered, distState.gs_on_tiles_offsets + tile_num - 1, sizeof(int), cudaMemcpyDeviceToHost);

		timer.start("23 updateDistributedStatLocally.getComputeLocally");
		// find the position by binary search or customized kernal function?
		// printf("dist_division_mode: %s, length: %d\n", dist_division_mode, strlen(dist_division_mode));
		if (strcmp(dist_division_mode, "rendered_num") == 0) {
			int num_rendered_per_device = num_rendered / world_size + 1;
			int last_local_num_rendered_end = num_rendered_per_device * local_rank;
			int local_num_rendered_end = min(num_rendered_per_device * (local_rank + 1), num_rendered);
			getComputeLocally <<<(tile_num + 255) / 256, 256 >>> (
				tile_num,
				distState.gs_on_tiles_offsets,
				distState.compute_locally,
				last_local_num_rendered_end,
				local_num_rendered_end
			);
			distState.last_local_num_rendered_end = last_local_num_rendered_end;
			distState.local_num_rendered_end = local_num_rendered_end;
		} else if (strcmp(dist_division_mode, "tile_num") == 0) {
			int num_tiles_per_device =	tile_num / world_size + 1;
			int last_local_num_rendered_end = num_tiles_per_device * local_rank;
			int local_num_rendered_end = min(num_tiles_per_device * (local_rank + 1), tile_num);
			//TODO: optimze this; in some cases, it will not be divied evenly -> 2170 will be into 1086 and 1084
			getComputeLocallyByTileNum <<<(tile_num + 255) / 256, 256 >>> (
				tile_num,
				distState.compute_locally,
				last_local_num_rendered_end,
				local_num_rendered_end
			);
			distState.last_local_num_rendered_end = last_local_num_rendered_end;
			distState.local_num_rendered_end = local_num_rendered_end;
		} else if (dist_division_mode[0] == 'T') {
			// dist_division_mode example: "T:0,1" or "T:10,20"
			char* dist_division_mode_left = new char[strlen(dist_division_mode) + 1];
			char* dist_division_mode_right = new char[strlen(dist_division_mode) + 1];
			strcpy(dist_division_mode_left, dist_division_mode);
			strcpy(dist_division_mode_right, dist_division_mode);
			
			char* pch = strtok(dist_division_mode_left, ":");
			pch = strtok(NULL, ":");
			pch = strtok(pch, ",");
			int tile_id_start = atoi(pch);
			pch = strtok(NULL, ",");
			int tile_id_end = atoi(pch);
			delete[] dist_division_mode_left;
			delete[] dist_division_mode_right;
			// printf("dist_division_mode is %s, tile_id_start is %d, tile_id_end is %d\n", dist_division_mode, tile_id_start, tile_id_end);
			
			getComputeLocallyByTileId <<<(tile_num + 255) / 256, 256 >>> (
				tile_num,
				distState.compute_locally,
				tile_id_start,
				tile_id_end
			);
			distState.last_local_num_rendered_end = tile_id_start;
			distState.local_num_rendered_end = tile_id_end;

		} else {
			// dist_division_mode example: "0,1" or "10,20"
			// TODO: refactor code: I should change it into: "R:0,1" or "R:10,20" later. refactor code.

			char* dist_division_mode_left = new char[strlen(dist_division_mode) + 1];
			char* dist_division_mode_right = new char[strlen(dist_division_mode) + 1];
			strcpy(dist_division_mode_left, dist_division_mode);
			strcpy(dist_division_mode_right, dist_division_mode);
			char* pch = strtok(dist_division_mode_left, ",");
			int row_id_start = atoi(pch);
			pch = strtok(NULL, ",");
			int row_id_end = atoi(pch);
			delete[] dist_division_mode_left;
			delete[] dist_division_mode_right;
			// printf("dist_division_mode is %s, row_id_start is %d, row_id_end is %d\n", dist_division_mode, row_id_start, row_id_end);

			getComputeLocallyByRowId <<<(tile_num + 255) / 256, 256 >>> (
				tile_num,
				distState.compute_locally,
				tile_grid.x,
				tile_grid.y,
				row_id_start,
				row_id_end
			);
			distState.last_local_num_rendered_end = row_id_start * tile_grid.x;
			distState.local_num_rendered_end = row_id_end * tile_grid.x;

			// printf("division_mode: %s is not supported.\n", dist_division_mode);
		}
		timer.stop("23 updateDistributedStatLocally.getComputeLocally");
	}
	else {
		cudaMemset(distState.compute_locally, true, tile_num * sizeof(bool));
	}
}

void save_log_in_file(int iteration, int local_rank, int world_size, std::string log_folder, const char* filename_prefix, const char* log_content) {
	char* filename = new char[64];
	sprintf(filename, "%s/%s_ws=%d_rk=%d.log", log_folder.c_str(), filename_prefix, world_size, local_rank);
	std::ofstream outfile;
	outfile.open(filename, std::ios_base::app);
	outfile << "iteration: " << iteration << ", " << log_content << "\n";
	outfile.close();
	delete[] filename;
}


/////////////////////////////// Preprocess ///////////////////////////////


// Define the function to process the dict and return a tuple with all variables
std::tuple<int, int, int, int, int, bool, bool, std::string, std::string, std::string>
 prepareArgs(const pybind11::dict &args) {
    std::string mode = args["mode"].cast<std::string>();
    std::string local_rank_str = args["local_rank"].cast<std::string>();
    std::string world_size_str = args["world_size"].cast<std::string>();
    std::string iteration_str = args["iteration"].cast<std::string>();
    std::string log_interval_str = args["log_interval"].cast<std::string>();
    std::string log_folder_str = args["log_folder"].cast<std::string>();
    std::string zhx_debug_str = args["zhx_debug"].cast<std::string>();
    std::string zhx_time_str = args["zhx_time"].cast<std::string>();
    std::string dist_division_mode_str = args["dist_division_mode"].cast<std::string>();

    int local_rank = std::stoi(local_rank_str);
    int world_size = std::stoi(world_size_str);
    int iteration = std::stoi(iteration_str);
    int log_interval = std::stoi(log_interval_str);
    bool zhx_debug = zhx_debug_str == "True";
    bool zhx_time = zhx_time_str == "True";

	int device;
	cudaError_t status = cudaGetDevice(&device);

	// Pack and return the variables in a tuple
    return std::make_tuple(local_rank, world_size, iteration, log_interval, device,
			zhx_debug, zhx_time,
			mode, dist_division_mode_str, log_folder_str);
}

// Forward rendering procedure for differentiable rasterization
// of Gaussians.
int CudaRasterizer::Rasterizer::preprocessForward(
	float2* means2D,
	float* depths,
	int* radii,
	float* cov3D,
	float4* conic_opacity,
	float* rgb,
	bool* clamped,//the above are all per-Gaussian intemediate results.
	const int P, int D, int M,
	const int width, int height,
	const float* means3D,
	const float* scales,
	const float* rotations,
	const float* shs,
	const float* opacities,//3dgs parameters
	const float scale_modifier,
	const float* viewmatrix,
	const float* projmatrix,
	const float* cam_pos,
	const float tan_fovx, float tan_fovy,
	const bool prefiltered,
	bool debug,//raster_settings
	const pybind11::dict &args)
{
	auto [local_rank, world_size, iteration, log_interval, device, zhx_debug, zhx_time, mode, dist_division_mode, log_folder] = prepareArgs(args);
	char* log_tmp = new char[500];

	// print out the environment variables
	if (mode == "train" && zhx_debug && iteration % log_interval == 1) {
		sprintf(log_tmp, "world_size: %d, local_rank: %d, iteration: %d, log_folder: %s, zhx_debug: %d, zhx_time: %d, device: %d, log_interval: %d, dist_division_mode: %s", 
				world_size, local_rank, iteration, log_folder.c_str(), zhx_debug, zhx_time, device, log_interval, dist_division_mode.c_str());
		save_log_in_file(iteration, local_rank, world_size, log_folder, "cuda", log_tmp);
	}

	MyTimerOnGPU timer;
	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

	dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	dim3 block(BLOCK_X, BLOCK_Y, 1);
	int tile_num = tile_grid.x * tile_grid.y;

	// allocate temporary buffer for tiles_touched.
	// In sep_rendering==True case, we will compute tiles_touched in the renderForward. 
	// TODO: remove it later by modifying FORWARD::preprocess when we deprecate sep_rendering==False case
	uint32_t* tiles_touched_temp_buffer;
	CHECK_CUDA(cudaMalloc(&tiles_touched_temp_buffer, P * sizeof(uint32_t)), debug);
	CHECK_CUDA(cudaMemset(tiles_touched_temp_buffer, 0, P * sizeof(uint32_t)), debug);

	timer.start("10 preprocess");
	// Run preprocessing per-Gaussian (transformation, bounding, conversion of SHs to RGB)
	CHECK_CUDA(FORWARD::preprocess(
		P, D, M,
		means3D,
		(glm::vec3*)scales,
		scale_modifier,
		(glm::vec4*)rotations,
		opacities,
		shs,
		clamped,
		nullptr,//cov3D_precomp,
		nullptr,//colors_precomp,TODO: this is correct?
		viewmatrix, projmatrix,
		(glm::vec3*)cam_pos,
		width, height,
		focal_x, focal_y,
		tan_fovx, tan_fovy,
		radii,
		means2D,
		depths,
		cov3D,
		rgb,
		conic_opacity,
		tile_grid,
		tiles_touched_temp_buffer,
		prefiltered,
		local_rank,
		world_size
	), debug)
	timer.stop("10 preprocess");

	int num_rendered = 0;//TODO: should I calculate this here?

	// Print out timing information
	if (zhx_time && iteration % log_interval == 1) {
		timer.printAllTimes(iteration, world_size, local_rank, log_folder, true);
	}
	delete log_tmp;
	// free temporary buffer for tiles_touched. TODO: remove it. 
	CHECK_CUDA(cudaFree(tiles_touched_temp_buffer), debug);
	return num_rendered;
}

void CudaRasterizer::Rasterizer::preprocessBackward(
	const int* radii,
	const float* cov3D,
	const bool* clamped,//the above are all per-Gaussian intemediate results.
	const int P, int D, int M, int R,
	const int width, int height,//rasterization setting.
	const float* means3D,
	const float* scales,
	const float* rotations,
	const float* shs,//input of this operator
	const float scale_modifier,
	const float* viewmatrix,
	const float* projmatrix,
	const float* campos,
	const float tan_fovx, float tan_fovy,//rasterization setting.
	const float* dL_dmean2D,
	const float* dL_dconic,
	float* dL_dcolor,//gradients of output of this operator. TODO: dL_dcolor is not const here because low-level implementation does not use const. Even though, we never modify it. 
	float* dL_dmean3D,
	float* dL_dcov3D,
	float* dL_dscale,
	float* dL_drot,
	float* dL_dsh,//gradients of input of this operator
	bool debug,
	const pybind11::dict &args)
{
	auto [local_rank, world_size, iteration, log_interval, device, zhx_debug, zhx_time, mode, dist_division_mode, log_folder] = prepareArgs(args);

	MyTimerOnGPU timer;
	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

	const float* cov3D_ptr = cov3D;
	timer.start("b20 preprocess");
	CHECK_CUDA(BACKWARD::preprocess(P, D, M,
		(float3*)means3D,
		radii,
		shs,
		clamped,
		(glm::vec3*)scales,
		(glm::vec4*)rotations,
		scale_modifier,
		cov3D_ptr,
		viewmatrix,
		projmatrix,
		focal_x, focal_y,
		tan_fovx, tan_fovy,
		(glm::vec3*)campos,
		(float3*)dL_dmean2D,
		dL_dconic,
		(glm::vec3*)dL_dmean3D,
		dL_dcolor,
		dL_dcov3D,
		dL_dsh,
		(glm::vec3*)dL_dscale,
		(glm::vec4*)dL_drot), debug)
	timer.stop("b20 preprocess");

	// Print out timing information
	if (zhx_time && iteration % log_interval == 1) {
		timer.printAllTimes(iteration, world_size, local_rank, log_folder, false);
	}
}





/////////////////////////////// GetDistributionStrategy ///////////////////////////////

void CudaRasterizer::Rasterizer::getDistributionStrategy(
	std::function<char* (size_t)> distBuffer,
	const int P,
	const int width, int height,
	float2* means2D,
	int* radii,
	bool* compute_locally,
	bool debug,
	const pybind11::dict &args)
{
	auto [local_rank, world_size, iteration, log_interval, device, zhx_debug, zhx_time, mode, dist_division_mode, log_folder] = prepareArgs(args);
	char* log_tmp = new char[500];

	MyTimerOnGPU timer;

	dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	int tile_num = tile_grid.x * tile_grid.y;

	size_t dist_chunk_size = required<DistributedState>(tile_grid.x * tile_grid.y);
	char* dist_chunkptr = distBuffer(dist_chunk_size);
	DistributedState distState = DistributedState::fromChunk(dist_chunkptr, tile_grid.x * tile_grid.y, true);

	distState.compute_locally = compute_locally;
	// NOTE: do not allocate memory for distState.compute_locally in fromChunk for sep_rendering mode, 
	// but use the compute_locally from python.

	// Use means2D and radii to decide how to evenly distribute the workloads. 
	timer.start("20 updateDistributedStatLocally");
	updateDistributedStatLocally(// FIXME: in memory_distribution mode, this function's calculation is not correct.
		P,
		width,
		height,
		tile_grid,
		radii,
		means2D,
		distState,
		local_rank,
		world_size,
		dist_division_mode.c_str(),
		timer
	);
	timer.stop("20 updateDistributedStatLocally");
	
	// DEBUG: print out compute_locally	information
	if (mode == "train" && zhx_debug && iteration % log_interval == 1) {
		int last_local_num_rendered_end = distState.last_local_num_rendered_end;
		int local_num_rendered_end = distState.local_num_rendered_end;
		uint32_t* gs_on_tiles_cpu = new uint32_t[tile_grid.x * tile_grid.y];
		CHECK_CUDA(cudaMemcpy(gs_on_tiles_cpu, distState.gs_on_tiles, tile_grid.x * tile_grid.y * sizeof(uint32_t), cudaMemcpyDeviceToHost), debug);

		// distState.compute_locally to cpu
		bool* compute_locally_cpu = new bool[tile_grid.x * tile_grid.y];
		CHECK_CUDA(cudaMemcpy(compute_locally_cpu, distState.compute_locally, tile_grid.x * tile_grid.y * sizeof(bool), cudaMemcpyDeviceToHost), debug);

		int num_local_tiles = 0;
		int local_tiles_left_idx = 999999999;
		int local_tiles_right_idx = 0;
		int num_rendered_from_distState = 0;
		for (int i = 0; i < tile_grid.x * tile_grid.y; i++)
		{
			if (compute_locally_cpu[i])
			{
				if (local_tiles_left_idx == 999999999)
					local_tiles_left_idx = i;
				local_tiles_right_idx = i;
				num_local_tiles++;
				num_rendered_from_distState += (int)gs_on_tiles_cpu[i];
			}
		}

		sprintf(log_tmp, "num_local_tiles: %d, local_tiles_left_idx: %d, local_tiles_right_idx: %d, last_local_num_rendered_end: %d, local_num_rendered_end: %d, num_rendered_from_distState: %d", 
			(int)num_local_tiles, (int)local_tiles_left_idx, (int)local_tiles_right_idx, (int)last_local_num_rendered_end, (int)local_num_rendered_end, (int)num_rendered_from_distState);
		save_log_in_file(iteration, local_rank, world_size, log_folder, "num_rendered", log_tmp);

		delete[] compute_locally_cpu;
		delete[] gs_on_tiles_cpu;
	}

	// Print out timing information
	if (zhx_time && iteration % log_interval == 1) {
		timer.printAllTimes(iteration, world_size, local_rank, log_folder, false);
	}

	delete[] log_tmp;
}

/////////////////////////////// Render ///////////////////////////////




int CudaRasterizer::Rasterizer::renderForward(
	std::function<char* (size_t)> geometryBuffer,
	std::function<char* (size_t)> binningBuffer,
	std::function<char* (size_t)> imageBuffer,
	const int P,
	const float* background,
	const int width, int height,
	float2* means2D,//TODO: do I have to add const for it? However, internal means2D is not const type. 
	float* depths,
	int* radii,
	float4* conic_opacity,
	float* rgb,
	bool* compute_locally,
	float* out_color,
	int* n_render,// TODO: int* could not match with uint32_t*. error may occur, especially when the number is large.
	int* n_consider,// If your uint32_t array contains values higher than 2,147,483,647, they will overflow when converted to int.
	int* n_contrib,//array of results for this function. 
	bool debug,
	const pybind11::dict &args)
{
	auto [local_rank, world_size, iteration, log_interval, device, zhx_debug, zhx_time, mode, dist_division_mode, log_folder] = prepareArgs(args);	
	char* log_tmp = new char[500];

	MyTimerOnGPU timer;

	size_t chunk_size = required<GeometryState>(P);
	char* chunkptr = geometryBuffer(chunk_size);
	GeometryState geomState = GeometryState::fromChunk(chunkptr, P, true); // do not allocate extra memory here if sep_rendering==True.

	dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	dim3 block(BLOCK_X, BLOCK_Y, 1);
	int tile_num = tile_grid.x * tile_grid.y;

	// Dynamically resize image-based auxiliary buffers during training
	size_t img_chunk_size = required<ImageState>(width * height);
	char* img_chunkptr = imageBuffer(img_chunk_size);
	ImageState imgState = ImageState::fromChunk(img_chunkptr, width * height);

	timer.start("24 updateDistributedStatLocally.updateTileTouched");
	// For sep_rendering==True case (here), we only compute tiles_touched in the renderForward.
	updateTileTouched <<<(P + 255) / 256, 256 >>> (
		P,
		tile_grid,
		radii,
		means2D,
		geomState.tiles_touched,
		compute_locally
	);
	timer.stop("24 updateDistributedStatLocally.updateTileTouched");

	// Compute prefix sum over full list of touched tile counts by Gaussians
	// E.g., [2, 3, 0, 2, 1] -> [2, 5, 5, 7, 8]
	timer.start("30 InclusiveSum");
	CHECK_CUDA(cub::DeviceScan::InclusiveSum(geomState.scanning_space, geomState.scan_size, geomState.tiles_touched, geomState.point_offsets, P), debug)
	timer.stop("30 InclusiveSum");

	// Retrieve total number of Gaussian instances to launch and resize aux buffers
	int num_rendered;
	CHECK_CUDA(cudaMemcpy(&num_rendered, geomState.point_offsets + P - 1, sizeof(int), cudaMemcpyDeviceToHost), debug);

	size_t binning_chunk_size = required<BinningState>(num_rendered);
	char* binning_chunkptr = binningBuffer(binning_chunk_size);
	BinningState binningState = BinningState::fromChunk(binning_chunkptr, num_rendered);

	timer.start("40 duplicateWithKeys");
	// For each instance to be rendered, produce adequate [ tile | depth ] key 
	// and corresponding dublicated Gaussian indices to be sorted
	duplicateWithKeys << <(P + 255) / 256, 256 >> > (
		P,
		means2D,
		depths,
		geomState.point_offsets,
		binningState.point_list_keys_unsorted,
		binningState.point_list_unsorted,
		radii,
		compute_locally,
		tile_grid,
		local_rank,
		world_size)
	CHECK_CUDA(, debug)
	timer.stop("40 duplicateWithKeys");

	int bit = getHigherMsb(tile_grid.x * tile_grid.y);

	timer.start("50 SortPairs");
	// Sort complete list of (duplicated) Gaussian indices by keys
	CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
		binningState.list_sorting_space,
		binningState.sorting_size,
		binningState.point_list_keys_unsorted, binningState.point_list_keys,
		binningState.point_list_unsorted, binningState.point_list,
		num_rendered, 0, 32 + bit), debug)
	timer.stop("50 SortPairs");

	CHECK_CUDA(cudaMemset(imgState.ranges, 0, tile_grid.x * tile_grid.y * sizeof(uint2)), debug);

	timer.start("60 identifyTileRanges");
	// Identify start and end of per-tile workloads in sorted list
	if (num_rendered > 0)
		identifyTileRanges << <(num_rendered + 255) / 256, 256 >> > (
			num_rendered,
			binningState.point_list_keys,
			imgState.ranges);
	CHECK_CUDA(, debug)
	timer.stop("60 identifyTileRanges");

	// Let each tile blend its range of Gaussians independently in parallel
	const float* feature_ptr = rgb;
	timer.start("70 render");
	CHECK_CUDA(FORWARD::render(//TODO: only deal with local tiles. do not even load other tiles.
		tile_grid, block,
		imgState.ranges,
		binningState.point_list,
		width, height,
		means2D,
		feature_ptr,
		conic_opacity,
		imgState.accum_alpha,
		imgState.n_contrib,
		imgState.n_contrib2loss,
		compute_locally,
		background,
		out_color), debug)
	timer.stop("70 render");

	// TODO: write a kernel to sum a block for n_contrib2loss and save the result in contrib. 
	// We may have different implementation.

	timer.start("81 sum_n_render");
	get_n_render<<< (tile_num + 255) / 256, 256 >>> (
		tile_num,
		imgState.ranges,
		n_render
	);
	timer.stop("81 sum_n_render");
	timer.start("82 sum_n_consider");
	reduce_data_per_block<< <tile_grid, block >> > (
		width, height,
		imgState.n_contrib,
		n_consider,
		compute_locally
	);
	timer.stop("82 sum_n_consider");
	timer.start("83 sum_n_contrib");
	reduce_data_per_block<< <tile_grid, block >> > (
		width, height,
		imgState.n_contrib2loss,
		n_contrib,
		compute_locally
	);
	timer.stop("83 sum_n_contrib");

	//////////////////////////// Logging && Save Statictis ////////////////////////////////////////////
	// DEBUG: print out timing information
	if (mode == "train" && zhx_time && iteration % log_interval == 1) {
		timer.printAllTimes(iteration, world_size, local_rank, log_folder, false);
	}

	// DEBUG: print out the number of Gaussians contributing to each pixel.
	if (mode == "train" && zhx_debug && iteration % log_interval == 1)
	{
		// move to imgState.ranges to cpu
		uint2* cpu_ranges = new uint2[tile_grid.x * tile_grid.y];
		CHECK_CUDA(cudaMemcpy(cpu_ranges, imgState.ranges, tile_grid.x * tile_grid.y * sizeof(uint2), cudaMemcpyDeviceToHost), debug);
		uint32_t* cpu_n_considered = new uint32_t[width * height];
		cudaMemcpy(cpu_n_considered, imgState.n_contrib, width * height * sizeof(uint32_t), cudaMemcpyDeviceToHost);
		uint32_t* cpu_n_contrib2loss = new uint32_t[width * height];
		cudaMemcpy(cpu_n_contrib2loss, imgState.n_contrib2loss, width * height * sizeof(uint32_t), cudaMemcpyDeviceToHost);

		float global_sum_n_rendered = 0;
		float global_sum_n_considered = 0;
		float global_sum_n_contrib2loss = 0;
		int total_pixels = 0;
		int num_local_tiles = 0;

		for (int i = 0; i < tile_grid.x * tile_grid.y; i++)
		{
			// output tile position and range
			int tile_x = i % tile_grid.x;
			int tile_y = i / tile_grid.x;
			int2 pix_min = { tile_x * BLOCK_X, tile_y * BLOCK_Y };
			int2 pix_max = { min(pix_min.x + BLOCK_X, width), min(pix_min.y + BLOCK_Y , height) };
			int num_pix = (pix_max.y - pix_min.y) * (pix_max.x - pix_min.x);
			int n_rendered = cpu_ranges[i].y - cpu_ranges[i].x;
			if (n_rendered <= 0)
				continue;

			int sum_n_considered = 0;
			int sum_n_contrib2loss = 0;
			for (int y = pix_min.y; y < pix_max.y; y++)
				for (int x = pix_min.x; x < pix_max.x; x++) {
					sum_n_considered += (int)cpu_n_considered[y * width + x];
					sum_n_contrib2loss += (int)cpu_n_contrib2loss[y * width + x];
				}
			float ave_n_considered = (float)sum_n_considered / num_pix;
			float ave_n_contrib2loss = (float)sum_n_contrib2loss / num_pix;

			float contrib2loss_ratio = 0;
			if (num_pix > 0)
				contrib2loss_ratio = ave_n_contrib2loss / n_rendered;

			sprintf(log_tmp, "tile: (%d, %d), range: (%d, %d), num_rendered_this_tile: %d, n_considered_per_pixel: %f, n_contrib2loss_per_pixel: %f, contrib2loss_ratio: %f", 
				tile_y,
				tile_x,
				(int)cpu_ranges[i].y,
				(int)cpu_ranges[i].x,
				n_rendered,
				ave_n_considered,
				ave_n_contrib2loss,
				contrib2loss_ratio);

			save_log_in_file(iteration, local_rank, world_size, log_folder, "n_contrib", log_tmp);
			global_sum_n_rendered += n_rendered;
			global_sum_n_considered += sum_n_considered;
			global_sum_n_contrib2loss += sum_n_contrib2loss;
			total_pixels += num_pix;
			num_local_tiles++;
		}
		float global_ave_n_rendered_per_pix = global_sum_n_rendered / (float)num_local_tiles;
		float global_ave_n_considered_per_pix = global_sum_n_considered / (float)total_pixels;
		float global_ave_n_contrib2loss_per_pix = global_sum_n_contrib2loss / (float)total_pixels;

		sprintf(log_tmp, "local_rank: %d, world_size: %d, num_tiles: %d, num_pixels: %d, num_rendered: %d, global_ave_n_rendered_per_pix: %f, global_ave_n_considered_per_pix: %f, global_ave_n_contrib2loss_per_pix: %f", 
			(int)local_rank,
			(int)world_size,
			(int)num_local_tiles,
			(int)total_pixels,
			(int)global_sum_n_rendered, 
			global_ave_n_rendered_per_pix, 
			global_ave_n_considered_per_pix, 
			global_ave_n_contrib2loss_per_pix
		);
		save_log_in_file(iteration, local_rank, world_size, log_folder, "n_contrib", log_tmp);

		delete[] cpu_ranges;
		delete[] cpu_n_considered;
		delete[] cpu_n_contrib2loss;
	}

	delete[] log_tmp;
	return num_rendered;
}

// Produce necessary gradients for optimization, corresponding
// to forward render pass
void CudaRasterizer::Rasterizer::renderBackward(
	const int P, int R,
	const float* background,
	const int width, int height,//rasterization settings. 
	char* geom_buffer,
	char* binning_buffer,
	char* img_buffer,
	bool* compute_locally,//buffer that contains intermedia results
	const float* dL_dpix,//gradient of output
	float* dL_dmean2D,//(P, 3)
	float* dL_dconic,
	float* dL_dopacity,
	float* dL_dcolor,//gradient of inputs
	const float2* means2D,
	const float4* conic_opacity,
	const float* rgb,//inputs
	bool debug,
	const pybind11::dict &args)
{
	auto [local_rank, world_size, iteration, log_interval, device, zhx_debug, zhx_time, mode, dist_division_mode, log_folder] = prepareArgs(args);

	MyTimerOnGPU timer;

	BinningState binningState = BinningState::fromChunk(binning_buffer, R);
	ImageState imgState = ImageState::fromChunk(img_buffer, width * height);

	const dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	const dim3 block(BLOCK_X, BLOCK_Y, 1);

	// Compute loss gradients w.r.t. 2D mean position, conic matrix,
	// opacity and RGB of Gaussians from per-pixel loss gradients.
	// If we were given precomputed colors and not SHs, use them.
	const float* color_ptr = rgb;
	timer.start("b10 render");
	CHECK_CUDA(BACKWARD::render(
		tile_grid,
		block,
		imgState.ranges,
		binningState.point_list,
		width, height,
		background,
		means2D,
		conic_opacity,
		color_ptr,
		imgState.accum_alpha,
		imgState.n_contrib,
		compute_locally,
		dL_dpix,
		(float3*)dL_dmean2D,
		(float4*)dL_dconic,
		dL_dopacity,
		dL_dcolor), debug)
	timer.stop("b10 render");

	// Print out timing information
	if (zhx_time && iteration % log_interval == 1) {
		timer.printAllTimes(iteration, world_size, local_rank, log_folder, false);
	}
}