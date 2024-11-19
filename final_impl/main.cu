#include "search-kernels.cuh"
#include "data.hpp"
#include <iostream>
#include <cassert>
#include "cudautils.cuh"
#include <vector>
#include <set>
#include <map>
#include <algorithm>
#include <chrono>

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

  char* pattern = argv[2];
  int p_size = ((std::string)pattern).size(); 
  int per_count=count_per(pattern);
  // ok cuda doesnt suppoer bitset so use uint
  std::vector<std::vector<uint64_t>> bitmasks=createBitmasks(pattern,p_size);    

  int* d_sizes, *d_matched_count;
  int* d_offsets;
  char* d_data;
  char* d_pattern;
  uint64_t* bitmasks1d;


  std::vector<uint64_t> h_flattened;
  std::vector<int> h_row_sizes;
    for (const auto& row : bitmasks) {
      h_row_sizes.push_back(row.size());
      h_flattened.insert(h_flattened.end(), row.begin(), row.end());
  }
  
  int total_rows = bitmasks.size();
  int total_elements = h_flattened.size();

  cudaMalloc(&bitmasks1d, total_elements * sizeof(uint64_t));

  cudaMalloc(&d_sizes, sizeof(int)*comments_column->size);
  cudaMalloc(&d_matched_count, sizeof(int));
  cudaMalloc(&d_offsets, sizeof(int)*comments_column->size);
  cudaMalloc(&d_data, sizeof(char)*data_size);
  cudaMalloc(&d_pattern, sizeof(char)*p_size);

  cudaMemcpy(bitmasks1d, h_flattened.data(), total_elements * sizeof(uint64_t), cudaMemcpyHostToDevice);
  cudaMemcpy(d_sizes, comments_column->sizes, sizeof(int)*comments_column->size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_offsets, comments_column->offsets, sizeof(int)*comments_column->size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_data, comments_column->data, sizeof(char)*data_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_pattern, pattern, sizeof(char)*p_size, cudaMemcpyHostToDevice);

  cudaMemset(d_matched_count, 0, sizeof(int));
  CUDACHKERR();

  int TB = 256;
  int gpu_matched_rows = 0;
  gpu_brute_force<<<std::ceil((float)comments_column->size/(float)TB), TB>>>(
    d_data, 
    d_offsets, 
    d_sizes, 
    comments_column->size, 
    d_matched_count,
    d_pattern, 
    p_size, 
    per_count, 
    bitmasks1d
  );
  
  CUDACHKERR();
  cudaMemcpy(&gpu_matched_rows, d_matched_count, sizeof(int), cudaMemcpyDeviceToHost);
  std::cout << "Result from GPU: " << gpu_matched_rows << "\n";
  CUDACHKERR();
}