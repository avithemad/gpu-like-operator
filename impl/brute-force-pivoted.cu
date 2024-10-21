#include "data.hpp"
#include <iostream>
#include <cassert>
#include "cudautils.cuh"

__global__ void gpu_brute_force_pivoted(char **data, int *max_lens, int size, char *pattern, int p_size, int *res)
{
  bool done = false;
  for (int i=0; i<max_lens[blockIdx.x] - p_size + 1; i++) {
    bool matched = true;
    for (int k=0; k<p_size; k++) {
      if (data[blockIdx.x][(i+k)*blockDim.x + threadIdx.x] != pattern[k]) {
        matched = false;
        // break;
      }
    }
    if (matched) {
      if (!done)
      atomicAdd(res, 1);
      done = true;
    }
  }
}
// kernel with early break optimization
__global__ void gpu_brute_force_pivoted_limited(char **data, int *max_lens, int size, char *pattern, int p_size, int *res)
{
  for (int i=0; i<max_lens[blockIdx.x] - p_size + 1; i++) {
    bool matched = true;
    for (int k=0; k<p_size; k++) {
      if (data[blockIdx.x][(i+k)*blockDim.x + threadIdx.x] != pattern[k]) {
        matched = false;
        break;
      }
    }
    if (matched) {
      atomicAdd(res, 1);
      break;
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
  if (comments_column == nullptr)
  {
    std::cout << "Unable to read comments columns, possibly no data in the file\n";
    exit(0);
  }

  int TB = 256;
  gpulike::StringColumnPivoted *comments_pivoted = gpulike::convert_to_transpose(comments_column, TB);
  // gpulike::print_pivoted_to_normal(comments_pivoted);
  const char* pattern = argv[2];
  char* d_pattern;
  int p_size = ((std::string)pattern).size();
  cudaMalloc(&d_pattern, sizeof(char)*p_size);
  cudaMemcpy(d_pattern, pattern, sizeof(char)*p_size, cudaMemcpyHostToDevice);
  char **d_data, **h_data;
  int *d_maxlens, *res, *res2;
  h_data = (char**)malloc(sizeof(char*)*comments_pivoted->size);
  char *d_char_data;
  int total_data = 0;
  for (int i=0; i<comments_pivoted->size; i++) total_data += comments_pivoted->max_lens[i];
  cudaMalloc(&d_char_data, sizeof(char)*total_data*TB);

  for (int i=0; i<comments_pivoted->size; i++) {
    h_data[i] = (i > 0 ? h_data[i-1] + comments_pivoted->max_lens[i-1]*TB : d_char_data);
    cudaMemcpy(h_data[i], comments_pivoted->data[i], sizeof(char)*comments_pivoted->max_lens[i]*TB, cudaMemcpyHostToDevice); 
  }
  cudaMalloc(&d_data, sizeof(char*)*comments_pivoted->size);
  cudaMemcpy(d_data, h_data, sizeof(char*)*comments_pivoted->size, cudaMemcpyHostToDevice);

  cudaMalloc(&d_maxlens, sizeof(int)*comments_pivoted->size);
  cudaMemcpy(d_maxlens, comments_pivoted->max_lens, sizeof(int)*comments_pivoted->size, cudaMemcpyHostToDevice);
  CUDACHKERR();

  cudaMalloc(&res, sizeof(int));
  cudaMalloc(&res2, sizeof(int));
  cudaMemset(res, 0, sizeof(int));
  cudaMemset(res2, 0, sizeof(int));

  gpu_brute_force_pivoted<<<comments_pivoted->size, TB>>>(d_data, d_maxlens, comments_pivoted->size, d_pattern, p_size, res);
  gpu_brute_force_pivoted_limited<<<comments_pivoted->size, TB>>>(d_data, d_maxlens, comments_pivoted->size, d_pattern, p_size, res2);

  CUDACHKERR();
  printf("Kernel done\n");
  // copy back results
  int h_res = 0;
  cudaMemcpy(&h_res, res, sizeof(int), cudaMemcpyDeviceToHost);
  std::cout << "GPU res: " << h_res << "\n";
}
