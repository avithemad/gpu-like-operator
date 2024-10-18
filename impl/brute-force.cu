#include "data.hpp"
#include <iostream>
#include <cassert>
#include "cudautils.cuh"

__global__ void gpu_brute_force(char* data, int* offsets, int* sizes, size_t table_size, char* pattern, int p_size, int* matched_count) {
  int tid = threadIdx.x + blockDim.x*blockIdx.x;
  if (tid >= table_size) return;
  // printf("GPU:%s\n", pattern);
  for (int j=0; j<sizes[tid] - p_size + 1; j++) {
    bool matched = true;
    for (int k=0; k<p_size; k++) {
      if (data[offsets[tid] + k + j] != pattern[k]) {
       matched = false;
       break; 
      }
    }
    if (matched) {
      atomicAdd(matched_count, 1);
      break;
    }
  }
}

int cpu_brute_force(gpulike::StringColumn* comments_column, std::string pattern) {
  int matched_rows = 0;
  // cpu side matching
  for (int i=0; i<comments_column->size; i++) {
    for (int j=0; j<(comments_column->sizes[i] - pattern.size() + 1); j++) {
      bool matched = true;
      for (int k=0; k<(pattern.size()); k++) {
        if (comments_column->data[comments_column->offsets[i]+j+k]!=pattern[k]) 
        {
          matched = false;
          break;  
        }

      }
      if (matched) {
        matched_rows++;
        break;
      }
    }
  }
  return matched_rows;
}

int main(int argc, char* argv[]) {
  if (argc < 3)
  {
    std::cout << "Please provide path to string column file and pattern. eg: ./brute-force /media/db/comments.txt <like-pattern>";
  }
  std::string txt_file = argv[1]; 

  gpulike::StringColumn* comments_column = gpulike::read_txt(txt_file);
  if (comments_column == nullptr) {
    std::cout << "Unable to read comments columns, possibly no data in the file\n";
    exit(0);
  }
  const std::string& main_string = comments_column->data;
  size_t data_size = 0;
  for (int i=0; i<comments_column->size; i++) data_size+=comments_column->sizes[i];

  std::cout << "Total rows: " <<  comments_column->size << "\n";

  const char* pattern = argv[2];
  int p_size = ((std::string)pattern).size();
  int cpu_matched_rows = cpu_brute_force(comments_column, pattern);
  std::cout << "Total matched rows in CPU: " << cpu_matched_rows << "\n";

  std::cout << "Now brute forcing in GPU\n"; 
  int* d_sizes, *d_matched_count;
  int* d_offsets;
  char* d_data;
  cudaMalloc(&d_sizes, sizeof(int)*comments_column->size);
  cudaMalloc(&d_matched_count, sizeof(int));
  cudaMalloc(&d_offsets, sizeof(int)*comments_column->size);
  cudaMalloc(&d_data, sizeof(char)*data_size);

  cudaMemcpy(d_sizes, comments_column->sizes, sizeof(int)*comments_column->size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_offsets, comments_column->offsets, sizeof(int)*comments_column->size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_data, comments_column->data, sizeof(char)*data_size, cudaMemcpyHostToDevice);
  cudaMemset(d_matched_count, 0, sizeof(int));
  CUDACHKERR();

  int TB = 32;
  char* d_pattern;
  cudaMalloc(&d_pattern, sizeof(char)*p_size);
  cudaMemcpy(d_pattern, pattern, sizeof(char)*p_size, cudaMemcpyHostToDevice);
  gpu_brute_force<<<std::ceil((float)comments_column->size/(float)TB), TB>>>(d_data, d_offsets, d_sizes, comments_column->size, d_pattern, p_size, d_matched_count);
  CUDACHKERR();
  int gpu_matched_rows = 0;
  cudaMemcpy(&gpu_matched_rows, d_matched_count, sizeof(int), cudaMemcpyDeviceToHost);
  CUDACHKERR();
  assert(gpu_matched_rows == cpu_matched_rows);

  std::cout << "Result from GPU: " << gpu_matched_rows << "\n";
}