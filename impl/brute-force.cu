#include "data.hpp"
#include <iostream>

void CUDACHKERR() {

  auto err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cout << "CUDA ERROR: " << cudaGetErrorString(err) << "\n"; 
  }
} 

__global__ void gpu_brute_force(char* data, char** addresses, int* sizes, size_t table_size, int* matched_count) {
  char* pattern = "packa";
  size_t p_size = 5;
  int tid = threadIdx.x + blockDim.x*blockIdx.x;
  if (tid >= table_size) return;
  // printf("GPU:%s\n", pattern);
  bool matched = false;
  for (int j=0; j<sizes[tid] - p_size; j++) {
    bool m = true;
    for (int k=0; k<p_size; k++) {
      if (addresses[tid][k+j] != pattern[k]) {
       m = false;
       break; 
      }
    }
    if (m) {
      matched = true;
      break;
    }
  }
  if (matched) {
    atomicAdd(matched_count, 1);
  }
}

int main() {
  std::string dbDir = "/media/ajayakar/space/src/tpch/data/tables/scale-1.0/";
  std::string lineitem_file = dbDir + "lineitem.parquet";

  auto lineitem_table = gpulike::getArrowTable(lineitem_file);

  // std::cout << lineitem_table->schema()->ToString() << "\n";

  gpulike::StringColumn* comments_column = gpulike::read_string_column(lineitem_table, "comments");

  const std::string& main_string = comments_column->data;
  size_t table_size = lineitem_table->num_rows();
  size_t data_size = 0;
  for (int i=0; i<table_size; i++) data_size+=comments_column->sizes[i];

  std::string pattern = "packa";
  int matched_rows = 0;
  // cpu side matching
  for (int i=0; i<table_size; i++) {
    bool m = false;
    for (int j=0; j<(comments_column->sizes[i] - pattern.size()); j++) {
      bool matched = true;
      for (int k=0; k<(pattern.size()); k++) {
        if (comments_column->stringAddresses[i][j+k]!=pattern[k]) 
        {
          matched = false;
          break;  
        }

      }
      if (matched) {
        m = true;
        break;
      }
    }
    if (m) {
      // std::cout << "Match found at row: " << i << "\n";
      matched_rows++;
      // std::cout << "\n";
    }
  }

  std::cout << "Total matched rows in CPU: " << matched_rows << "\n";

  std::cout << "Now brute forcing in GPU\n"; 
  int* d_sizes, *d_matched_count;
  char** d_addresses;
  char* d_data;
  cudaMalloc(&d_sizes, sizeof(int)*table_size);
  cudaMalloc(&d_matched_count, sizeof(int));
  cudaMalloc(&d_addresses, sizeof(char*)*table_size);
  cudaMalloc(&d_data, sizeof(char)*data_size);

  int z = 0;
  cudaMemcpy(d_sizes, comments_column->sizes, sizeof(int)*table_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_addresses, comments_column->stringAddresses, sizeof(char*)*table_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_matched_count, &z, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_data, comments_column->data, sizeof(char)*data_size, cudaMemcpyHostToDevice);
  CUDACHKERR();
  int TB = 32;
  gpu_brute_force<<<std::ceil(table_size/TB), TB>>>(d_data, d_addresses, d_sizes, table_size, d_matched_count);
  CUDACHKERR();
  cudaMemcpy(&z, d_matched_count, sizeof(int), cudaMemcpyDeviceToHost);
  CUDACHKERR();

  std::cout << "Result from GPU: " << z << "\n";
}