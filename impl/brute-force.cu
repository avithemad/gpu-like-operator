#include "data.hpp"
#include <iostream>
#include <cassert>

void CUDACHKERR() {

  auto err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cout << "CUDA ERROR: " << cudaGetErrorString(err) << "\n"; 
  }
} 

__global__ void gpu_brute_force(char* data, int* offsets, int* sizes, size_t table_size, int* matched_count) {
  char* pattern = "packa";
  size_t p_size = 5;
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

int cpu_brute_force(int table_size, gpulike::StringColumn* comments_column, std::string pattern) {
  int matched_rows = 0;
  // cpu side matching
  for (int i=0; i<table_size; i++) {
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

  if (argc < 2) {
    std::cout << "Please provide path to string column file. eg: ./brute-force /media/db/comments.txt";
  }
  std::string txt_file = argv[1]; 

  gpulike::StringColumn* comments_column = gpulike::read_txt(txt_file);
  if (comments_column == nullptr) {
    std::cout << "Unable to read comments columns, possibly no data in the file\n";
    exit(0);
  }
  const std::string& main_string = comments_column->data;
  size_t table_size = gpulike::read_txt_size(txt_file);
  size_t data_size = 0;
  for (int i=0; i<table_size; i++) data_size+=comments_column->sizes[i];

  std::cout << "Total rows: " <<  table_size << "\n";

  std::string pattern = "packa";
  int cpu_matched_rows = cpu_brute_force(table_size, comments_column, "packa");
  std::cout << "Total matched rows in CPU: " << cpu_matched_rows << "\n";

  std::cout << "Now brute forcing in GPU\n"; 
  int* d_sizes, *d_matched_count;
  int* d_offsets;
  char* d_data;
  cudaMalloc(&d_sizes, sizeof(int)*table_size);
  cudaMalloc(&d_matched_count, sizeof(int));
  cudaMalloc(&d_offsets, sizeof(int)*table_size);
  cudaMalloc(&d_data, sizeof(char)*data_size);

  cudaMemcpy(d_sizes, comments_column->sizes, sizeof(int)*table_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_offsets, comments_column->offsets, sizeof(int)*table_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_data, comments_column->data, sizeof(char)*data_size, cudaMemcpyHostToDevice);
  cudaMemset(d_matched_count, 0, sizeof(int));
  CUDACHKERR();

  int TB = 32;
  gpu_brute_force<<<std::ceil((float)table_size/(float)TB), TB>>>(d_data, d_offsets, d_sizes, table_size, d_matched_count);
  CUDACHKERR();
  int gpu_matched_rows = 0;
  cudaMemcpy(&gpu_matched_rows, d_matched_count, sizeof(int), cudaMemcpyDeviceToHost);
  CUDACHKERR();
  assert(gpu_matched_rows == cpu_matched_rows);

  std::cout << "Result from GPU: " << gpu_matched_rows << "\n";
}