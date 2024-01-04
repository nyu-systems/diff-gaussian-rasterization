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
#include "my_timer.cu"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <string>
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

CudaRasterizer::GeometryState CudaRasterizer::GeometryState::fromChunk(char*& chunk, size_t P)
{
	GeometryState geom;
	obtain(chunk, geom.depths, P, 128);
	obtain(chunk, geom.clamped, P * 3, 128);
	obtain(chunk, geom.internal_radii, P, 128);
	obtain(chunk, geom.means2D, P, 128);
	obtain(chunk, geom.cov3D, P * 6, 128);
	obtain(chunk, geom.conic_opacity, P, 128);
	obtain(chunk, geom.rgb, P * 3, 128);
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
		binning.point_list_unsorted, binning.point_list, P);//TODO: why do we need to sort here? so weird. 
	obtain(chunk, binning.list_sorting_space, binning.sorting_size, 128);
	return binning;
}

CudaRasterizer::DistributedState CudaRasterizer::DistributedState::fromChunk(char*& chunk, size_t tile_num)
{
	DistributedState dist;
	obtain(chunk, dist.gs_on_tiles, tile_num, 128);
	obtain(chunk, dist.gs_on_tiles_offsets, tile_num, 128);
	cub::DeviceScan::InclusiveSum(nullptr, dist.scan_size, dist.gs_on_tiles, dist.gs_on_tiles_offsets, tile_num);
	obtain(chunk, dist.scanning_space, dist.scan_size, 128);
	obtain(chunk, dist.compute_locally, tile_num, 128);
	return dist;
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

void updateDistributedStatLocally(//TODO: optimize implementations for all these kernels. 
	const int P,
	const int width,
	const int height,
	const dim3 tile_grid,
	int* radii,
	float2* means2D,
	uint32_t* tiles_touched,
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
		} else {
			printf("division_mode: %s is not supported.\n", dist_division_mode);
		}
		timer.stop("23 updateDistributedStatLocally.getComputeLocally");


	}
	else {
		cudaMemset(distState.compute_locally, true, tile_num * sizeof(bool));
	}

	timer.start("24 updateDistributedStatLocally.updateTileTouched");
	// set tiles_touched[i] to 0 if compute_locally[i] is false.
	updateTileTouched <<<(P + 255) / 256, 256 >>> (
		P,
		tile_grid,
		radii,
		means2D,
		tiles_touched,
		distState.compute_locally
	);
	timer.stop("24 updateDistributedStatLocally.updateTileTouched");
}

void save_log_in_file(int iteration, int local_rank, int world_size, const char* log_folder, const char* filename_prefix, const char* log_content) {
	char* filename = new char[100];
	sprintf(filename, "%s/%s_ws=%d_rk=%d.log", log_folder, filename_prefix, world_size, local_rank);
	std::ofstream outfile;
	outfile.open(filename, std::ios_base::app);
	outfile << "iteration: " << iteration << ", " << log_content << "\n";
	outfile.close();
	delete[] filename;
}

// Forward rendering procedure for differentiable rasterization
// of Gaussians.
int CudaRasterizer::Rasterizer::forward(
	std::function<char* (size_t)> geometryBuffer,
	std::function<char* (size_t)> binningBuffer,
	std::function<char* (size_t)> imageBuffer,
	std::function<char* (size_t)> distBuffer,
	const int P, int D, int M,
	const float* background,
	const int width, int height,
	const float* means3D,
	const float* shs,
	const float* colors_precomp,
	const float* opacities,
	const float* scales,
	const float scale_modifier,
	const float* rotations,
	const float* cov3D_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const float* cam_pos,
	const float tan_fovx, float tan_fovy,
	const bool prefiltered,
	float* out_color,
	int* radii,
	bool debug)
{
	int local_rank = get_env_var("LOCAL_RANK");
	int world_size = get_env_var("WORLD_SIZE");
	if (world_size == 0) world_size = 1;
	int iteration = get_env_var("ITERATION");
	int log_interval = get_env_var("LOG_INTERVAL");
	const char* log_folder = getenv("LOG_FOLDER");
	const char* zhx_debug_str = getenv("ZHX_DEBUG");
	bool zhx_debug = false;
	if (zhx_debug_str != nullptr && strcmp(zhx_debug_str, "true") == 0) zhx_debug = true;
	const char* zhx_time_str = getenv("ZHX_TIME");
	bool zhx_time = false;
	if (zhx_time_str != nullptr && strcmp(zhx_time_str, "true") == 0) zhx_time = true;
	char* log_tmp = new char[500];
	const char* dist_division_mode = getenv("DIST_DIVISION_MODE");
	// print out the environment variables
	int device;
	cudaError_t status = cudaGetDevice(&device);

	if (zhx_debug && iteration % log_interval == 1) {
		// convert zhx_debug, zhx_time, device into one char string for output.
		sprintf(log_tmp, "world_size: %d, local_rank: %d, iteration: %d, log_folder: %s, zhx_debug: %d, zhx_time: %d, device: %d", world_size, local_rank, iteration, log_folder, zhx_debug, zhx_time, device);
		save_log_in_file(iteration, local_rank, world_size, log_folder, "cuda", log_tmp);
	}

	// MyTimer timer;
	MyTimerOnGPU timer;//TODO: two types of timer.
	timer.start("00 forward");

	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

	size_t chunk_size = required<GeometryState>(P);
	char* chunkptr = geometryBuffer(chunk_size);
	GeometryState geomState = GeometryState::fromChunk(chunkptr, P);

	if (radii == nullptr)
	{
		radii = geomState.internal_radii;
	}

	dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	dim3 block(BLOCK_X, BLOCK_Y, 1);

	// Dynamically resize image-based auxiliary buffers during training
	size_t img_chunk_size = required<ImageState>(width * height);
	char* img_chunkptr = imageBuffer(img_chunk_size);
	ImageState imgState = ImageState::fromChunk(img_chunkptr, width * height);

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
		geomState.clamped,
		cov3D_precomp,
		colors_precomp,
		viewmatrix, projmatrix,
		(glm::vec3*)cam_pos,
		width, height,
		focal_x, focal_y,
		tan_fovx, tan_fovy,
		radii,
		geomState.means2D,
		geomState.depths,
		geomState.cov3D,
		geomState.rgb,
		geomState.conic_opacity,
		tile_grid,
		geomState.tiles_touched,
		prefiltered,
		local_rank,
		world_size
	), debug)
	timer.stop("10 preprocess");

	size_t dist_chunk_size = required<DistributedState>(tile_grid.x * tile_grid.y);
	char* dist_chunkptr = distBuffer(dist_chunk_size);
	DistributedState distState = DistributedState::fromChunk(dist_chunkptr, tile_grid.x * tile_grid.y);
	// Use geomState.means2D and radii to decide how to evenly distribute the workloads. 
	timer.start("20 updateDistributedStatLocally");
	if (world_size >= 1) {
		updateDistributedStatLocally(
			P,
			width,
			height,
			tile_grid,
			radii,
			geomState.means2D,
			geomState.tiles_touched,
			distState,
			local_rank,
			world_size,
			dist_division_mode,
			timer
		);
	} else {
		int tile_num = tile_grid.x * tile_grid.y;
		cudaMemset(distState.compute_locally, true, tile_num * sizeof(bool));
	}
	timer.stop("20 updateDistributedStatLocally");

	// CHECK_CUDA(, debug)
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
		geomState.means2D,
		geomState.depths,
		geomState.point_offsets,
		binningState.point_list_keys_unsorted,
		binningState.point_list_unsorted,
		radii,
		distState.compute_locally,
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

	// DEBUG: print out rectangle and touched information for each tile.
	if (false && iteration % log_interval == 1)// TODO: set different debug levels.
	{
		if (world_size == 1)
		{
			// move radii, geomState.means2D back to cpu.
			int* radii_cpu = new int[P];
			CHECK_CUDA(cudaMemcpy(radii_cpu, radii, P * sizeof(int), cudaMemcpyDeviceToHost), debug);
			float2* means2D_cpu = new float2[P];
			CHECK_CUDA(cudaMemcpy(means2D_cpu, geomState.means2D, P * sizeof(float2), cudaMemcpyDeviceToHost), debug);

			std::ofstream fout;
			char* filename = new char[100];
			log_folder = log_folder == nullptr ? "logs" : log_folder;
			sprintf(filename, "%s/rectangle_iter=%d.txt", log_folder, iteration);
			fout.open(filename, std::ios::app);

			int number_rendered_tmp = 0;
			for (int i = 0; i < P; i++) {
				// save radii_cpu, means2D_cpu to file.
				uint2 rect_min, rect_max;
				if (radii_cpu[i]>0) // if radii_cpu[i] is 0 which means (in_frustum == false), then do not consider it. 
					getRect(means2D_cpu[i], radii_cpu[i], rect_min, rect_max, tile_grid);
				else {
					rect_min.x = 0;
					rect_min.y = 0;
					rect_max.x = 0;
					rect_max.y = 0;
				}
				number_rendered_tmp += (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
				fout << means2D_cpu[i].x << " " << means2D_cpu[i].y << " " << radii_cpu[i] << " " << rect_min.x << " " << rect_min.y << " " << rect_max.x << " " << rect_max.y << "\n";
			}
			fout << "number of rendered: (1, new calculated) " << number_rendered_tmp << " (2, from cuda code) " << num_rendered << "\n";
			// clean up
			fout.close();
			delete[] radii_cpu;
			delete[] means2D_cpu;
			delete[] filename;
		}

		// move to imgState.ranges to cpu
		uint2* cpu_ranges = new uint2[tile_grid.x * tile_grid.y];
		CHECK_CUDA(cudaMemcpy(cpu_ranges, imgState.ranges, tile_grid.x * tile_grid.y * sizeof(uint2), cudaMemcpyDeviceToHost), debug);

		sprintf(log_tmp, "iteration: %d, local_rank: %d, world_size: %d, num_rendered: %d, grid: (%d, %d)", iteration, local_rank, world_size, num_rendered, tile_grid.x, tile_grid.y);
		save_log_in_file(iteration, local_rank, world_size, log_folder, "ranges", log_tmp);

		// DEBUG: print out tile ranges
		// for (int i = 0; i < tile_grid.x * tile_grid.y; i++)
		// {
		// 	// output tile position and range
		// 	outfile << i << ": (" << i % tile_grid.x << "," << i / tile_grid.x << "), (x,y)=(" << cpu_ranges[i].x << "," << cpu_ranges[i].y << ")\n";
		// }
		delete[] cpu_ranges;
	}

	// Let each tile blend its range of Gaussians independently in parallel
	const float* feature_ptr = colors_precomp != nullptr ? colors_precomp : geomState.rgb;
	timer.start("70 render");
	CHECK_CUDA(FORWARD::render(//TODO: only deal with local tiles. do not even load other tiles.
		tile_grid, block,
		imgState.ranges,
		binningState.point_list,
		width, height,
		geomState.means2D,
		feature_ptr,
		geomState.conic_opacity,
		imgState.accum_alpha,
		imgState.n_contrib,
		imgState.n_contrib2loss,
		distState.compute_locally,
		background,
		out_color), debug)
	timer.stop("70 render");

	timer.stop("00 forward");

	// DEBUG: print out timing information
	if (zhx_time && iteration % log_interval == 1) {
		timer.printAllTimes(iteration, world_size, local_rank, log_folder);
	}

	// DEBUG: print out compute_locally	information
	if (zhx_debug && iteration % log_interval == 1) {
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

		sprintf(log_tmp, "iteration: %d, num_local_tiles: %d, local_tiles_left_idx: %d, local_tiles_right_idx: %d, last_local_num_rendered_end: %d, local_num_rendered_end: %d, num_rendered: %d, num_rendered_from_distState: %d", (int)iteration, (int)num_local_tiles, (int)local_tiles_left_idx, (int)local_tiles_right_idx, (int)last_local_num_rendered_end, (int)local_num_rendered_end, (int)num_rendered, (int)num_rendered_from_distState);
		save_log_in_file(iteration, local_rank, world_size, log_folder, "num_rendered", log_tmp);

		delete[] compute_locally_cpu;
		delete[] gs_on_tiles_cpu;
	}

	// DEBUG: print out the number of Gaussians contributing to each pixel.
	if (zhx_debug && iteration % log_interval == 1)
	{
		// move to imgState.ranges to cpu
		uint2* cpu_ranges = new uint2[tile_grid.x * tile_grid.y];
		CHECK_CUDA(cudaMemcpy(cpu_ranges, imgState.ranges, tile_grid.x * tile_grid.y * sizeof(uint2), cudaMemcpyDeviceToHost), debug);
		uint32_t* cpu_n_contrib = new uint32_t[width * height];
		cudaMemcpy(cpu_n_contrib, imgState.n_contrib, width * height * sizeof(uint32_t), cudaMemcpyDeviceToHost);
		uint32_t* cpu_n_contrib2loss = new uint32_t[width * height];
		cudaMemcpy(cpu_n_contrib2loss, imgState.n_contrib2loss, width * height * sizeof(uint32_t), cudaMemcpyDeviceToHost);

		float global_n_contrib = 0;
		float global_n_contrib_ratio = 0;
		float global_n_contrib2loss = 0;
		int num_local_tiles = 0;

		for (int i = 0; i < tile_grid.x * tile_grid.y; i++)
		{
			// output tile position and range
			int tile_x = i % tile_grid.x;
			int tile_y = i / tile_grid.x;
			int2 pix_min = { tile_x * BLOCK_X, tile_y * BLOCK_Y };
			int2 pix_max = { min(pix_min.x + BLOCK_X, width), min(pix_min.y + BLOCK_Y , height) };
			int sum_n_contrib = 0;
			int sum_n_contrib2loss = 0;
			for (int y = pix_min.y; y < pix_max.y; y++)
				for (int x = pix_min.x; x < pix_max.x; x++) {
					sum_n_contrib += (int)cpu_n_contrib[y * width + x];
					sum_n_contrib2loss += (int)cpu_n_contrib2loss[y * width + x];
				}
			float ave_n_contrib = (float)sum_n_contrib / ((pix_max.y - pix_min.y) * (pix_max.x - pix_min.x));
			float ave_n_contrib2loss = (float)sum_n_contrib2loss / ((pix_max.y - pix_min.y) * (pix_max.x - pix_min.x));

			float contrib_ratio = 0;
			if (cpu_ranges[i].x < cpu_ranges[i].y)
				contrib_ratio = ave_n_contrib2loss / (cpu_ranges[i].y - cpu_ranges[i].x);

			sprintf(log_tmp, "tile: (%d, %d), range: (%d, %d), local_num_rendered: %d, local_last_n_contrib: %f, local_real_n_contrib: %f, contrib_ratio: %f", tile_y, tile_x, (int)cpu_ranges[i].y, (int)cpu_ranges[i].x, (int)cpu_ranges[i].y-(int)cpu_ranges[i].x, ave_n_contrib, ave_n_contrib2loss, contrib_ratio);
			save_log_in_file(iteration, local_rank, world_size, log_folder, "n_contrib", log_tmp);
			global_n_contrib += ave_n_contrib;
			global_n_contrib2loss += ave_n_contrib2loss;
			global_n_contrib_ratio += contrib_ratio;
			if (cpu_ranges[i].x < cpu_ranges[i].y) num_local_tiles++;
		}
		global_n_contrib = global_n_contrib / (float)num_local_tiles;
		global_n_contrib2loss = global_n_contrib2loss / (float)num_local_tiles;
		global_n_contrib_ratio = global_n_contrib_ratio / (float)num_local_tiles;
		sprintf(log_tmp, "iteration: %d, local_rank: %d, world_size: %d, num_local_tiles: %d, global_num_rendered: %d, global_last_n_contrib: %f, global_n_contrib2loss: %f, contrib_ratio: %f", iteration, local_rank, world_size, num_local_tiles, num_rendered, global_n_contrib, global_n_contrib2loss, global_n_contrib_ratio);
		save_log_in_file(iteration, local_rank, world_size, log_folder, "n_contrib", log_tmp);

		delete[] cpu_ranges;
		delete[] cpu_n_contrib;
	}

	delete[] log_tmp;
	return num_rendered;
}

// Produce necessary gradients for optimization, corresponding
// to forward render pass
void CudaRasterizer::Rasterizer::backward(
	const int P, int D, int M, int R,
	const float* background,
	const int width, int height,
	const float* means3D,
	const float* shs,
	const float* colors_precomp,
	const float* scales,
	const float scale_modifier,
	const float* rotations,
	const float* cov3D_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const float* campos,
	const float tan_fovx, float tan_fovy,
	const int* radii,
	char* geom_buffer,
	char* binning_buffer,
	char* img_buffer,
	char* dist_buffer,
	const float* dL_dpix,
	float* dL_dmean2D,
	float* dL_dconic,
	float* dL_dopacity,
	float* dL_dcolor,
	float* dL_dmean3D,
	float* dL_dcov3D,
	float* dL_dsh,
	float* dL_dscale,
	float* dL_drot,
	bool debug)
{
	int local_rank = get_env_var("LOCAL_RANK");
	int world_size = get_env_var("WORLD_SIZE");
	if (world_size == 0) world_size = 1;
	int iteration = get_env_var("ITERATION");
	const char* log_folder = getenv("LOG_FOLDER");
	int log_interval = get_env_var("LOG_INTERVAL");
	const char* zhx_debug_str = getenv("ZHX_DEBUG");
	bool zhx_debug = false;
	if (zhx_debug_str != nullptr && strcmp(zhx_debug_str, "true") == 0) zhx_debug = true;
	const char* zhx_time_str = getenv("ZHX_TIME");
	bool zhx_time = false;
	if (zhx_time_str != nullptr && strcmp(zhx_time_str, "true") == 0) zhx_time = true;

	// MyTimer timer;
	MyTimerOnGPU timer;//TODO: two types of timer.
	timer.start("b00 backward");

	GeometryState geomState = GeometryState::fromChunk(geom_buffer, P);
	BinningState binningState = BinningState::fromChunk(binning_buffer, R);
	ImageState imgState = ImageState::fromChunk(img_buffer, width * height);

	if (radii == nullptr)
	{
		radii = geomState.internal_radii;
	}

	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

	const dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	const dim3 block(BLOCK_X, BLOCK_Y, 1);

	DistributedState distState = DistributedState::fromChunk(dist_buffer, tile_grid.x * tile_grid.y);

	// Compute loss gradients w.r.t. 2D mean position, conic matrix,
	// opacity and RGB of Gaussians from per-pixel loss gradients.
	// If we were given precomputed colors and not SHs, use them.
	const float* color_ptr = (colors_precomp != nullptr) ? colors_precomp : geomState.rgb;
	timer.start("b10 render");
	CHECK_CUDA(BACKWARD::render(
		tile_grid,
		block,
		imgState.ranges,
		binningState.point_list,
		width, height,
		background,
		geomState.means2D,
		geomState.conic_opacity,
		color_ptr,
		imgState.accum_alpha,
		imgState.n_contrib,
		distState.compute_locally,
		dL_dpix,
		(float3*)dL_dmean2D,
		(float4*)dL_dconic,
		dL_dopacity,
		dL_dcolor), debug)
	timer.stop("b10 render");

	// Take care of the rest of preprocessing. Was the precomputed covariance
	// given to us or a scales/rot pair? If precomputed, pass that. If not,
	// use the one we computed ourselves.
	const float* cov3D_ptr = (cov3D_precomp != nullptr) ? cov3D_precomp : geomState.cov3D;
	timer.start("b20 preprocess");
	CHECK_CUDA(BACKWARD::preprocess(P, D, M,
		(float3*)means3D,
		radii,
		shs,
		geomState.clamped,
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
	timer.stop("b00 backward");

	// Print out timing information
	if (zhx_time && iteration % log_interval == 1) {
		timer.printAllTimes(iteration, world_size, local_rank, log_folder);
	}
}