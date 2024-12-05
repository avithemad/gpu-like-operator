#include "brute_force_pivoted.cuh"
#include "kmp_step_pivoted.cuh"
#include "kmp_basic_pivoted.cuh"
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

  gpulike::StringColumn* column = gpulike::read_txt(txt_file);
  std::cout << "text file read\n";
  if (column == nullptr) {
    std::cout << "Unable to read comments columns, possibly no data in the file\n";
    exit(0);
  }
  gpulike::StringColumnPivotedK *pivoted_col = gpulike::convert_to_pivotedk(column);
  std::cout << "pivoted conversion done\n";

  free(column->data);
  free(column->offsets);

  std::cout << "Total rows: " <<  pivoted_col->size << "\n";

  // std::vector<std::string> pats = {"%ry%", "%nd%", "%en%", "%es", "%ly%", "%express%dependencies%", "%requests%ly%"};
  std::vector<std::string> pats = {"%es%", "%requests%ly%"};
  // std::vector<std::string> pats = {"%express%dependencies%", "%requests%ly%"};

  int *d_matched_count, *d_sp_sizes, *d_sizes;
  int* d_prefix_table, *d_prefix_table_sizes;
  int64_t *d_wildcard_bm;
  char* d_data;
  char* d_pattern;
  cudaMalloc(&d_sizes, sizeof(int)*pivoted_col->size);
  cudaMalloc(&d_data, sizeof(char)*pivoted_col->max_len*pivoted_col->size);
  cudaMalloc(&d_matched_count, sizeof(int));
  cudaMemcpy(d_sizes, column->sizes, sizeof(int)*pivoted_col->size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_data, pivoted_col->data, sizeof(char)*pivoted_col->max_len*pivoted_col->size, cudaMemcpyHostToDevice);

  for (auto pat: pats) {
    char* pattern = (char*)malloc(sizeof(char) * pat.size() );
    for (int i=0; i<pat.size(); i++) pattern[i] = pat[i];


    int p_size = pat.size(); 

  // preprocess the pattern
  auto preprocessed_pattern = preprocess_pattern(pattern, p_size);
  preprocessed_pattern.print();


  cudaMalloc(&d_pattern, sizeof(char)*p_size);
  cudaMalloc(&d_sp_sizes, sizeof(int)*preprocessed_pattern.sp_sizes.size());
  cudaMalloc(&d_wildcard_bm, sizeof(int64_t)*preprocessed_pattern.wildcards.size());
  cudaMalloc(&d_prefix_table, sizeof(int)*preprocessed_pattern.prefix_tables_gpu.size());
  cudaMalloc(&d_prefix_table_sizes, sizeof(int)*preprocessed_pattern.prefix_tables_gpu_sizes.size());

  cudaMemcpy(d_sp_sizes, preprocessed_pattern.sp_sizes.data(), 
    sizeof(int)*preprocessed_pattern.sp_sizes.size(), 
    cudaMemcpyHostToDevice);
  cudaMemcpy(d_wildcard_bm, preprocessed_pattern.wildcards.data(), 
    sizeof(int64_t)*preprocessed_pattern.wildcards.size(), 
    cudaMemcpyHostToDevice);
  cudaMemcpy(d_pattern, preprocessed_pattern.pattern.c_str(), 
    sizeof(char)*preprocessed_pattern.pattern.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(d_prefix_table, preprocessed_pattern.prefix_tables_gpu.data(), 
    sizeof(int)*preprocessed_pattern.prefix_tables_gpu.size(), 
    cudaMemcpyHostToDevice);
  cudaMemcpy(d_prefix_table_sizes, preprocessed_pattern.prefix_tables_gpu_sizes.data(), 
    sizeof(int)*preprocessed_pattern.prefix_tables_gpu_sizes.size(), 
    cudaMemcpyHostToDevice);

  CUDACHKERR();

  // int TB = 256;
  std::vector<int> TBs = {256};
  for (auto TB: TBs) {
    std::cout << "Using thread block size: " << TB << std::endl;
    int gpu_matched_rows = 0;
    cudaMemset(d_matched_count, 0, sizeof(int));
    gpu_brute_force_pivoted<<<std::ceil((float)column->size/(float)TB), TB>>>(
      d_data, 
      d_sizes,
      pivoted_col->size, 
      d_matched_count,
      d_pattern, 
      preprocessed_pattern.pattern.size(), 
      d_sp_sizes,
      preprocessed_pattern.sp_sizes.size(),
      d_wildcard_bm,
      preprocessed_pattern.wildcards.size()
      ,d_prefix_table,
      d_prefix_table_sizes
    );
    CUDACHKERR();
    cudaMemcpy(&gpu_matched_rows, d_matched_count, sizeof(int), cudaMemcpyDeviceToHost);
    CUDACHKERR();
    std::cout << "BRUTE FORCE: " << gpu_matched_rows << "\n";

    cudaMemset(d_matched_count, 0, sizeof(int));
    gpu_brute_force_pivoted_step<<<std::ceil((float)column->size/(float)TB), TB>>>(
      d_data, 
      d_sizes,
      pivoted_col->size, 
      d_matched_count,
      d_pattern, 
      preprocessed_pattern.pattern.size(), 
      d_sp_sizes,
      preprocessed_pattern.sp_sizes.size(),
      d_wildcard_bm,
      preprocessed_pattern.wildcards.size()
      ,d_prefix_table,
      d_prefix_table_sizes
    );
    CUDACHKERR();
    cudaMemcpy(&gpu_matched_rows, d_matched_count, sizeof(int), cudaMemcpyDeviceToHost);
    CUDACHKERR();
    std::cout << "BRUTE FORCE: " << gpu_matched_rows << "\n";

    cudaMemset(d_matched_count, 0, sizeof(int));
    gpu_kmp_basic_pivoted<<<std::ceil((float)column->size/(float)TB), TB>>>(
      d_data,  
      d_sizes, 
      column->size, 
      d_matched_count,
      d_pattern, 
      preprocessed_pattern.pattern.size(), 
      d_sp_sizes,
      preprocessed_pattern.sp_sizes.size(),
      d_wildcard_bm,
      preprocessed_pattern.wildcards.size()
      ,d_prefix_table,
      d_prefix_table_sizes
    );
    CUDACHKERR();
    cudaMemcpy(&gpu_matched_rows, d_matched_count, sizeof(int), cudaMemcpyDeviceToHost);
    CUDACHKERR();
    std::cout << "KMP BASIC: " << gpu_matched_rows << "\n";

    cudaMemset(d_matched_count, 0, sizeof(int));
    gpu_kmp_step_pivoted<<<std::ceil((float)column->size/(float)TB), TB>>>(
      d_data, 
      d_sizes, 
      pivoted_col->size, 
      d_matched_count,
      d_pattern, 
      preprocessed_pattern.pattern.size(), 
      d_sp_sizes,
      preprocessed_pattern.sp_sizes.size(),
      d_wildcard_bm,
      preprocessed_pattern.wildcards.size()
      ,d_prefix_table,
      d_prefix_table_sizes
    );
    CUDACHKERR();
    cudaMemcpy(&gpu_matched_rows, d_matched_count, sizeof(int), cudaMemcpyDeviceToHost);
    CUDACHKERR();
    std::cout << "KMP STEP: " << gpu_matched_rows << "\n";
  }
  free(pattern);
  }
}