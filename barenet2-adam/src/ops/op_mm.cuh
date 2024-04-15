// #pragma once

// #include "utils/assert.cuh"
// #include "utils/tensor.cuh"

// #define MM_BLOCK_SIZE 32
// // This operator compute C = A@B
// template <typename T>
// void op_mm(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &C)
// {
//   assert(A.h == C.h && B.w == C.w && A.w == B.h);
//   assert(A.on_device && B.on_device && C.on_device);

//   // Lab-1: please complete this
//   // You need to define separate kernel function(s) and launch them here
//   // delete assert(0) when you are finished

//   Tensor<T> A_d = A.toDevice();
//   Tensor<T> B_d = B.toDevice();
//   Tensor<T> C_d = C.toDevice();
//   dim3 dimBlock(MM_BLOCK_SIZE, MM_BLOCK_SIZE);

//   dim3 dimGrid((C.h + MM_BLOCK_SIZE - 1) / MM_BLOCK_SIZE, (C.w + MM_BLOCK_SIZE - 1) / MM_BLOCK_SIZE);
//   op_mm_kernel<<<1, 1>>>(A_d, B_d, C_d);
// }

// template <typename T>
// __global__ void op_mm_kernel(Tensor<T> A, Tensor<T> B, Tensor<T> C)
// {
//   // Lab-1: add your code here

//   int block_h_start = blockIdx.x * blockDim.x;
//   // int block_h_end = min((blockIdx.x+1)*blockDim.x, A.h);

//   int block_w_start = blockIdx.y * blockDim.y;
//   // int block_w_end = min((blockIdx.y+1)*blockDim.y, A.w);

//   int row = threadIdx.x;
//   int col = threadIdx.y;

//   float r = 0.0;
//   // int sub_size_h = min(MM_BLOCK_SIZE, A.h - block_h_start);
//   // int sub_size_w = min(MM_BLOCK_SIZE, A.w - block_w_start);
//   __shared__ float As[MM_BLOCK_SIZE][MM_BLOCK_SIZE];
//   __shared__ float Bs[MM_BLOCK_SIZE][MM_BLOCK_SIZE];

//   int block_row = block_h_start + row;
//   int block_col = block_w_start + col;

//  for (; block_row < A.h; block_row += blockDim.x * gridDim.x)
//  {
//    for (; block_col < A.w; block_col += blockDim.y * gridDim.y)
//    {

//   for (int k = 0; k < (A.w + MM_BLOCK_SIZE - 1) / MM_BLOCK_SIZE; k++)
//   {
//     int k_start = k * MM_BLOCK_SIZE;
//     // int k_end = (k+1)*MM_BLOCK_SIZE;

//     // Tensor<T> ASub = A.slice(block_h_start, block_h_end, k_start, k_end);
//     // Tensor<T> BSub = B.slice(k_start, k_end, block_w_start, block_w_end);
//     if (block_row < A.h && k_start + col < A.w)
//       As[row][col] = Index(A, block_row, k_start + col);
//     if (!IndexOutofBound(B, k_start + row, block_col))
//       Bs[row][col] = Index(B, k_start + row, block_col);
//     __syncthreads();

//   if (block_row < C.h && block_col < C.w) {
//     for (int e = 0; e < MM_BLOCK_SIZE; e++)
//     {
//       r += As[row][e] * Bs[e][col];
//     }
//   }
//     __syncthreads();
//   }
//   if (block_row < C.h && block_col < C.w)
//     Index(C, block_row, block_col) = r;
//   }
//   }
//   // for (int i = 0; i < A.h; i++) {
//   //     for (int j = 0; j < B.w; j++) {
//   //         for (int k = 0; k < A.w; k++) {
//   //             Index(C, i, j) += Index(A, i, k) * Index(B, k, j);
//   //         }
//   //     }
//   // }
// }

#pragma once

#include "utils/assert.cuh"
#include "utils/tensor.cuh"

#define MM_BLOCK_SIZE 32
// This operator compute C = A@B
template <typename T>
void op_mm(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &C)
{
  assert(A.h == C.h && B.w == C.w && A.w == B.h);
  assert(A.on_device && B.on_device && C.on_device);

  // Lab-1: please complete this
  // You need to define separate kernel function(s) and launch them here
  // delete assert(0) when you are finished

  Tensor<T> A_d = A.toDevice();
  Tensor<T> B_d = B.toDevice();
  Tensor<T> C_d = C.toDevice();

  if (C.h == 1 && C.w == 1)
  {
    // 1xn @ nx1 = 1x1
    op_inner_prod_kernel<<<1, 1>>>(A_d, B_d, C_d);
  }
  else if (C.h == 1 && C.w != 1)
  {
    // 1xm @ mxn = 1xn
    int dimBlock = min(1024, C.w);
    op_xTA_prod_kernel<<<1, dimBlock>>>(A_d, B_d, C_d);
  }
  else if (A.h != 1 && A.w == 1)
  {
    // mx1 @ 1xn = mxn
    dim3 dimBlock(min(MM_BLOCK_SIZE, C.h), min(MM_BLOCK_SIZE, C.w));
    dim3 dimGrid((C.h + MM_BLOCK_SIZE - 1) / MM_BLOCK_SIZE, (C.w + MM_BLOCK_SIZE - 1) / MM_BLOCK_SIZE);
    op_xTy_prod_kernel<<<dimGrid, dimBlock>>>(A_d, B_d, C_d);
  }
  else if (A.h != 1 && A.w != 1 && C.w == 1)
  {
    // mxn @ nx1 = mx1
    int dimBlock = min(1024, A.h);
    op_Ax_prod_kernel<<<1, dimBlock>>>(A_d, B_d, C_d);
  }
  else
  {
    dim3 dimBlock(MM_BLOCK_SIZE, MM_BLOCK_SIZE);
    dim3 dimGrid((C.h + MM_BLOCK_SIZE - 1) / MM_BLOCK_SIZE, (C.w + MM_BLOCK_SIZE - 1) / MM_BLOCK_SIZE);
    op_mm_kernel<<<dimGrid, dimBlock>>>(A_d, B_d, C_d);
  }
}

template <typename T>
__global__ void op_inner_prod_kernel(Tensor<T> A, Tensor<T> B, Tensor<T> C)
{
  // 1xn @ nx1 = 1x1
  // <<<1, 1>>>
  float r = 0.0;
  for (int i = 0; i < A.w; i++)
  {
    r += Index(A, 0, i) * Index(B, i, 0);
  }
  Index(C, 0, 0) = r;
}

template <typename T>
__global__ void op_xTA_prod_kernel(Tensor<T> A, Tensor<T> B, Tensor<T> C)
{
  // 1xm @ mxn = 1xn
  // <<<1, n>>>
  int n = threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (; n < C.w; n += stride)
  {
    float r = 0.0;
    for (int j = 0; j < A.w; j++)
    {
      r += Index(A, 0, j) * Index(B, j, n);
    }
    Index(C, 0, n) = r;
  }
}

template <typename T>
__global__ void op_xTy_prod_kernel(Tensor<T> A, Tensor<T> B, Tensor<T> C)
{
  // mx1 @ 1xn = mxn
  // <<<m, n>>>
  int row = blockIdx.x * blockDim.x + threadIdx.x; // row number in C
  int col = blockIdx.y * blockDim.y + threadIdx.y; // col number in C
  int stride_x = blockDim.x * gridDim.x;
  int stride_y = blockDim.y * gridDim.y;
  for (; row < C.h; row += stride_x) //  + MM_BLOCK_SIZE - 1 ensures all threads enter
  {
    for (; col < C.w; col += stride_y){
      Index(C, row, col) = Index(A, row, 0) * Index(B, 0, col);
    }
  }
}

template <typename T>
__global__ void op_Ax_prod_kernel(Tensor<T> A, Tensor<T> B, Tensor<T> C)
{
  // mxn @ nx1 = mx1
  // <<<1, m>>>
  int m = threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (; m < C.h; m += stride)
  {
    float r = 0.0;
    for (int j = 0; j < A.w; j++)
    {
      r += Index(A, m, j) * Index(B, j, 0);
    }
    Index(C, m, 0) = r;
  }
}

template <typename T>
__global__ void op_mm_kernel(Tensor<T> A, Tensor<T> B, Tensor<T> C)
{
  // Lab-1: add your code here

  int row = blockIdx.x * blockDim.x + threadIdx.x; // row number in C
  int col = blockIdx.y * blockDim.y + threadIdx.y; // col number in C
  int stride_x = blockDim.x * gridDim.x;
  int stride_y = blockDim.y * gridDim.y;
  int block_x = threadIdx.x; // x offset in block (shared)
  int block_y = threadIdx.y; // y offset in block (shared)

  float r = 0.0;

  for (; row < C.h + MM_BLOCK_SIZE - 1; row += stride_x) //  + MM_BLOCK_SIZE - 1 ensures all threads enter
  {
    for (; col < C.w + MM_BLOCK_SIZE - 1; col += stride_y)
    {

      __shared__ float As[MM_BLOCK_SIZE][MM_BLOCK_SIZE];
      __shared__ float Bs[MM_BLOCK_SIZE][MM_BLOCK_SIZE];

      #pragma unroll
      for (int i = 0; i < (A.w + MM_BLOCK_SIZE - 1) / MM_BLOCK_SIZE; i++)
      {
        int tile_start = i * MM_BLOCK_SIZE;  // col offset for current tile in A (row in B)
        int tile_row = tile_start + block_x; // row offset for current element in B
        int tile_col = tile_start + block_y; // col offset for current element in A
        // read data from global memory to shared memory
        if (!IndexOutofBound(A, row, tile_col))
        {
          As[block_x][block_y] = Index(A, row, tile_col);
        }
        else
        {
          As[block_x][block_y] = 0.0;
        }

        if (!IndexOutofBound(B, tile_row, col))
        {
          Bs[block_x][block_y] = Index(B, tile_row, col);
        }
        else
        {
          Bs[block_x][block_y] = 0.0;
        }
        __syncthreads();

        if (!IndexOutofBound(C, row, col))
        {
          #pragma unroll
          for (int j = 0; j < min(MM_BLOCK_SIZE, A.w - tile_start); j++)
          {
            r += As[block_x][j] * Bs[j][block_y];
          }
        }

        __syncthreads();
      }

      if (!IndexOutofBound(C, row, col))
      {
        Index(C, row, col) = r;
      }
    }
  }
}