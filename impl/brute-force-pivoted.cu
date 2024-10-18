#include "data.hpp"
#include <iostream>
#include <cassert>
#include "cudautils.cuh"

__global__ void gpu_brute_force_pivoted(char **data, int *max_lens, int size, char *pattern, int p_size, int *res)
{
  // int tid = threadIdx.x + blockDim.x*blockIdx.x;
  // if (tid >= size*blockDim.x) return; // do you really need this check??
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
  if (argc < 2)
  {
    std::cout << "Please provide path to string column file. eg: ./brute-force /media/db/comments.txt";
  }
  std::string txt_file = argv[1];

  gpulike::StringColumn *comments_column = gpulike::read_txt(txt_file);
  if (comments_column == nullptr)
  {
    std::cout << "Unable to read comments columns, possibly no data in the file\n";
    exit(0);
  }

  int TB = 32;
  gpulike::StringColumnPivoted *comments_pivoted = gpulike::convert_to_transpose(comments_column, TB);
  // gpulike::print_pivoted_to_normal(comments_pivoted);
  const char* pattern = "packa";
  char* d_pattern;
  int p_size = 5;
  cudaMalloc(&d_pattern, sizeof(char)*p_size);
  cudaMemcpy(d_pattern, pattern, sizeof(char)*p_size, cudaMemcpyHostToDevice);
  char **d_data, **h_data;
  int *d_maxlens, *res;
  h_data = (char**)malloc(sizeof(char*)*comments_pivoted->size);
  for (int i=0; i<comments_pivoted->size; i++) {
    cudaMalloc(&h_data[i], sizeof(char)*comments_pivoted->max_lens[i]*TB);
    cudaMemcpy(h_data[i], comments_pivoted->data[i], sizeof(char)*comments_pivoted->max_lens[i]*TB, cudaMemcpyHostToDevice); 
  CUDACHKERR();
  }
  cudaMalloc(&d_data, sizeof(char*)*comments_pivoted->size);
  cudaMemcpy(d_data, h_data, sizeof(char*)*comments_pivoted->size, cudaMemcpyHostToDevice);

  CUDACHKERR();
  cudaMalloc(&d_maxlens, sizeof(int)*comments_pivoted->size);
  cudaMemcpy(d_maxlens, comments_pivoted->max_lens, sizeof(int)*comments_pivoted->size, cudaMemcpyHostToDevice);
  CUDACHKERR();

  cudaMalloc(&res, sizeof(int));
  cudaMemset(res, 0, sizeof(int));

  gpu_brute_force_pivoted<<<comments_pivoted->size, TB>>>(d_data, d_maxlens, comments_pivoted->size, d_pattern, 
    p_size, res);

  CUDACHKERR();
  // copy back results
  int h_res = 0;
  cudaMemcpy(&h_res, res, sizeof(int), cudaMemcpyDeviceToHost);
  std::cout << "GPU res: " << h_res << "\n";
}
