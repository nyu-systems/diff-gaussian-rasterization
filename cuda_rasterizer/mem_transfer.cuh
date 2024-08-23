#include "cuda_runtime.h"
#include <numeric>
#include <chrono>
#include <algorithm>
#include <vector>
#include <cub/cub.cuh> // Why including cub make compilation fail?

__global__ void get_rank2id(bool *mask, int *mask_presum, int *rank2id, int n_row) {
    int n_threads = gridDim.x * blockDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_row; i += n_threads) {
        if (mask[i]) {
            rank2id[mask_presum[i]-1] = i;
        }
    }
}

// Transfer attr to scattered dest.
__global__ void kernel(
    float *h_attr_1,
    float *h_attr_2,
    float *h_attr_3,
    float *h_attr_4,
    float *h_attr_5,
    int M1,
    int M2,
    int M3,
    int M4,
    int M5,
    int *rank2id,
    int num_select,
    float *d_dest_1,
    float *d_dest_2,
    float *d_dest_3,
    float *d_dest_4,
    float *d_dest_5
) {
    int stride = gridDim.x * blockDim.x;
    int total_elements = num_select * (M1 + M2 + M3 + M4 + M5);
    int offset;
    int offset_source;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total_elements; i += stride) {
        if (i < num_select * M1) {
            offset = i;
            offset_source = rank2id[offset / M1] * M1 + offset % M1;
            d_dest_1[offset] = h_attr_1[offset_source];
        }
        else if (i < num_select * (M1 + M2)) {
            offset = i - num_select * M1;
            offset_source = rank2id[offset / M2] * M2 + offset % M2;
            d_dest_2[offset] = h_attr_2[offset_source];
        }
        else if (i < num_select * (M1 + M2 + M3)) {
            offset = i - num_select * (M1 + M2);
            offset_source = rank2id[offset / M3] * M3 + offset % M3;
            d_dest_3[offset] = h_attr_3[offset_source];
        }
        else if (i < num_select * (M1 + M2 + M3 + M4)) {
            offset = i - num_select * (M1 + M2 + M3);
            offset_source = rank2id[offset / M4] * M4 + offset % M4;
            d_dest_4[offset] = h_attr_4[offset_source];
        }
        else {
            offset = i - num_select * (M1 + M2 + M3 + M4);
            offset_source = rank2id[offset / M5] * M5 + offset % M5;
            d_dest_5[offset] = h_attr_5[offset_source];
        }
    }
}

void scattered_transfer(
    float *h_attr_1,
    float *h_attr_2,
    float *h_attr_3,
    float *h_attr_4,
    float *h_attr_5,
    bool *d_mask,
    int M1,
    int M2,
    int M3,
    int M4,
    int M5,
    int N,
    int num_select,
    float *d_dest_1,
    float *d_dest_2,
    float *d_dest_3,
    float *d_dest_4,
    float *d_dest_5
) {
    // calculate rank2id
    void     *d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0;
    int      *d_mask_presum;
    cudaMalloc(&d_mask_presum, N * sizeof(int));
    cudaMemset(d_mask_presum, 0, N * sizeof(int));
    // cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_mask, d_mask_presum, N);
    // cudaMalloc(&d_temp_storage, temp_storage_bytes);
    // cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_mask, d_mask_presum, N);

    int *d_rank2id;
    cudaMalloc(&d_rank2id, num_select * sizeof(int));
    get_rank2id<<<64, 256>>>(d_mask, d_mask_presum, d_rank2id, N);
    cudaDeviceSynchronize();
    
    // launch kernel
    int block_size = 256;
    int grid_size = 32;

    kernel<<<grid_size, block_size>>>(
        h_attr_1,
        h_attr_2,
        h_attr_3,
        h_attr_4,
        h_attr_5,
        M1,
        M2,
        M3,
        M4,
        M5,
        d_rank2id,
        num_select,
        d_dest_1,
        d_dest_2,
        d_dest_3,
        d_dest_4,
        d_dest_5
    );

    cudaFree(d_mask_presum);
    cudaFree(d_temp_storage);
}
