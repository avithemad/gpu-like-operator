#include "data.hpp"
#include <iostream>
#include <cassert>
#include "cudautils.cuh"

__global__ void gpu_brute_force_pivotedk(char **data, int max_len, int size, char *pattern, int p_size, int *res)
{
  int tid = threadIdx.x + blockDim.x*blockIdx.x;
  // printf("%d\n", tid);
  if (tid >= size) return;
  // #pragma unroll(4)
  for (int i=0; i<max_len - p_size + 1; i+=4) {
    bool matched = true;
    
    for (int k=0; k<p_size; k++) {
      if (data[i + k][tid] != pattern[k]) {
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
  gpulike::StringColumnPivotedK *comments_pivoted = gpulike::convert_to_pivotedk(comments_column);

  const char* pattern = argv[2];
  int p_size = ((std::string)pattern).size();
  char* d_pattern;
  cudaMalloc(&d_pattern, sizeof(char)*p_size);
  cudaMemcpy(d_pattern, pattern, sizeof(char)*p_size, cudaMemcpyHostToDevice);

  char *d_char_data, **h_data;
  cudaMalloc(&d_char_data, sizeof(char)*comments_pivoted->max_len*comments_pivoted->size);
  h_data = (char**)malloc(sizeof(char*)*comments_pivoted->max_len);
  for (int i=0; i<comments_pivoted->max_len; i++) {
    h_data[i] = (i > 0 ? h_data[i-1] + comments_pivoted->size : d_char_data);
    cudaMemcpy(h_data[i], comments_pivoted->data[i], sizeof(char)*comments_pivoted->size, cudaMemcpyHostToDevice);
  }
  char **d_data;
  cudaMalloc(&d_data, sizeof(char*)*comments_pivoted->max_len);
  cudaMemcpy(d_data, h_data, sizeof(char*)*comments_pivoted->max_len, cudaMemcpyHostToDevice);
  CUDACHKERR();
  int *res;
  cudaMalloc(&res, sizeof(int));
  cudaMemset(res, 0, sizeof(int));

  gpu_brute_force_pivotedk<<<std::ceil((float)comments_pivoted->size/(float)TB), TB>>>(
    d_data, comments_pivoted->max_len, comments_pivoted->size, d_pattern, p_size, res
  );

  CUDACHKERR();
  int gpu_matched_rows = 0;
  cudaMemcpy(&gpu_matched_rows, res, sizeof(int), cudaMemcpyDeviceToHost);
  std::cout << "Result from GPU: " << gpu_matched_rows << "\n";
}
