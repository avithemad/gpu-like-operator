#include <cufft.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include "cudautils.cuh"
#include "data.hpp"
#include <set>

#define CHUNK_SIZE 64

  __global__ void multiplyFFT(float2 *p1, float2 *p2, float2 *res)
  {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    res[tid].x = p1[tid].x * p2[threadIdx.x].x - p1[tid].y * p2[threadIdx.x].y;
    res[tid].y = p1[tid].y * p2[threadIdx.x].x + p1[tid].x * p2[threadIdx.x].y;
  }

__global__ void preparePrefix(float2 *p1, float *prefix, int size)
{
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid >= size)
    return;
  for (int i = 1; i <= CHUNK_SIZE; i++)
  {
    prefix[tid * (CHUNK_SIZE + 1) + i] = prefix[tid * (CHUNK_SIZE + 1) + i - 1] + 
    (p1[tid * (CHUNK_SIZE ) + i - 1].x * p1[tid * (CHUNK_SIZE) + i - 1].x);
  }
}

__global__ void normalize(float2 *res, int size) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid >= size) return;

  res[tid].x /= CHUNK_SIZE;
}

__global__ void obtainFilterMask(float2 *convolution, float *prefix, float pattern_sq, int size, int pattern_size,
                                 bool *res, int *text_sizes)
{
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid >= size)
    return;

  for (int i = 0; i <= text_sizes[tid] - pattern_size + 1; i++)
  {
    int resl = prefix[tid * (CHUNK_SIZE+1) + pattern_size + i] -
               prefix[tid * (CHUNK_SIZE+1) + i] + pattern_sq - 2 * convolution[tid * CHUNK_SIZE + i + pattern_size - 1].x;

    if (resl == 0)
    {
      res[tid] = 1;
      return;
    }
  }
}

int main(int argc, char *argv[])
{
  if (argc < 3)
  {
    std::cout << "Please provide path to string column file and pattern. eg: ./brute-force /media/db/comments.txt <like-pattern>";
  }
  std::string txt_file = argv[1];

  gpulike::StringColumn *comments_column = gpulike::read_txt(txt_file);
  std::cout << "total comments size: " << comments_column->size << std::endl;
  char *pattern = argv[2];
  size_t pattern_size = ((std::string)pattern).size();

  // first try it row by row dft
  /**
   * assumptions:
   * 1. datatype is varchar(256)
   * 2. pattern length is less than 256.
   *
   * design space
   * - floating point precision fp16 vs fp32
   * - assumption on the datatype
   */
  // prepare data for fft ready
  float2 *text_poly = (float2 *)malloc(sizeof(float2) * comments_column->size * CHUNK_SIZE);
  memset(text_poly, sizeof(float2) * CHUNK_SIZE * comments_column->size, 0);

  int max_text = 0;
  for (int i = 0; i < comments_column->size; i++)
  {
    for (int j = 0; j < comments_column->sizes[i]; j++)
    {
      text_poly[CHUNK_SIZE * i + j].x = comments_column->data[comments_column->offsets[i] + j];
    }
    max_text = std::max(max_text, comments_column->sizes[i]);
  }
  std::cout << "max length of the text: " << max_text << std::endl;
  std::vector<float2> pattern_poly(CHUNK_SIZE, {0.0, 0.0});
  for (int i = 0; i < pattern_size; i++)
  {
    pattern_poly[i].x = pattern[(pattern_size-1) - i];
  }
  // preparation for the data is done

  // do the convolution
  float2 *d_poly1, *d_poly2, *d_result;
  float2 *d_poly1_copy;
  cudaCheckErrors(cudaMalloc(&d_poly1, sizeof(float2) * CHUNK_SIZE * comments_column->size));
  cudaCheckErrors(cudaMalloc(&d_poly1_copy, sizeof(float2) * CHUNK_SIZE * comments_column->size));
  cudaCheckErrors(cudaMalloc(&d_poly2, sizeof(float2) * CHUNK_SIZE * comments_column->size));
  cudaCheckErrors(cudaMalloc(&d_result, sizeof(float2) * CHUNK_SIZE * comments_column->size));
  cudaCheckErrors(cudaMemset(d_result, 0, sizeof(float2) * CHUNK_SIZE * comments_column->size));
  cudaCheckErrors(cudaMalloc(&d_poly2, sizeof(float2) * CHUNK_SIZE));

  cudaCheckErrors(cudaMemcpy(d_poly1, text_poly, sizeof(float2) * CHUNK_SIZE * comments_column->size, cudaMemcpyHostToDevice));
  cudaCheckErrors(cudaMemcpy(d_poly1_copy, text_poly, sizeof(float2) * CHUNK_SIZE * comments_column->size, cudaMemcpyHostToDevice));
  cudaCheckErrors(cudaMemcpy(d_poly2, pattern_poly.data(), sizeof(float2) * CHUNK_SIZE, cudaMemcpyHostToDevice));
  cufftHandle plan_forward_text, plan_forward_pattern, plan_inverse_text;
  cufftPlan1d(&plan_forward_text, CHUNK_SIZE, CUFFT_C2C, comments_column->size);
  cufftPlan1d(&plan_forward_pattern, CHUNK_SIZE, CUFFT_C2C, 1);
  CUDACHKERR();

  cufftExecC2C(plan_forward_text, d_poly1, d_poly1, CUFFT_FORWARD);
  cufftExecC2C(plan_forward_pattern, d_poly2, d_poly2, CUFFT_FORWARD);
  CUDACHKERR();
  cudaDeviceSynchronize();
  multiplyFFT<<<comments_column->size, CHUNK_SIZE>>>(d_poly1, d_poly2, d_result);
  CUDACHKERR();

  cufftPlan1d(&plan_inverse_text, CHUNK_SIZE, CUFFT_C2C, comments_column->size);
  cufftExecC2C(plan_inverse_text, d_result, d_result, CUFFT_INVERSE);
  CUDACHKERR();
  normalize<<<std::ceil((float)(CHUNK_SIZE*comments_column->size)/(float)256), 256>>>(
    d_result, comments_column->size*CHUNK_SIZE
  );

  float pattern_sq = 0.;
  for (auto p : pattern_poly)
    pattern_sq += pow(p.x, 2);
  float *d_poly1_prefix;
  cudaCheckErrors(cudaMalloc(&d_poly1_prefix, sizeof(float) * (CHUNK_SIZE + 1) * comments_column->size));
  preparePrefix<<<std::ceil(float(comments_column->size) / 256), 256>>>(d_poly1_copy, d_poly1_prefix, comments_column->size);
  int *d_text_sizes;
  bool *d_bitmask_result;
  cudaCheckErrors(cudaMalloc(&d_text_sizes, sizeof(int) * comments_column->size));
  cudaCheckErrors(cudaMalloc(&d_bitmask_result, sizeof(bool) * comments_column->size));
  cudaMemset(d_bitmask_result, 0, sizeof(bool) * comments_column->size);
  cudaCheckErrors(cudaMemcpy(d_text_sizes, comments_column->sizes, sizeof(int) * comments_column->size, cudaMemcpyHostToDevice));
  obtainFilterMask<<<std::ceil(float(comments_column->size) / 256), 256>>>(
      d_result,
      d_poly1_prefix,
      pattern_sq,
      comments_column->size,
      pattern_size,
      d_bitmask_result,
      d_text_sizes);
  bool *bitmask_result = (bool *)malloc(sizeof(bool) * comments_column->size);
  cudaMemcpy(bitmask_result, d_bitmask_result, sizeof(bool) * comments_column->size, cudaMemcpyDeviceToHost);
  int res = 0;
  for (int i = 0; i < comments_column->size; i++) {

    res += bitmask_result[i];
    // if (bitmask_result[i]) {
    //   std::cout << i+1 << "\n";
    // }
  }
  std::cout << "total matches: " << res << std::endl;
}