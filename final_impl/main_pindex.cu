#include "brute_force.cuh"
#include "brute_force_step.cuh"
#include "kmp_basic.cuh"
#include "kmp_step.cuh"
#include "pindex.cuh"
#include "data.hpp"
#include <iostream>
#include <cassert>
#include "cudautils.cuh"
#include <vector>
#include <set>
#include <map>
#include <algorithm>
#include <chrono>


char to_lower(char c)
{
  if ((c >= 'A' && c <= 'Z'))
    return (c - 'A') + 'a';
  else
  {
    return c;
  } 
}

int main(int argc, char* argv[]) {
  if (argc < 3)
  {
    std::cout << "Please provide path to string column file and pattern. eg: ./brute-force /media/db/comments.txt <like-pattern>";
    exit(0);
  }
  std::string txt_file = argv[1]; 

  gpulike::StringColumn* column = gpulike::read_txt(txt_file);
  std::cout << "text file read\n";
  if (column == nullptr) {
    std::cout << "Unable to read comments columns, possibly no data in the file\n";
    exit(0);
  }
  // gpulike::StringColumnPivotedK *pivoted_col = gpulike::convert_to_pivotedk(column);
  // std::cout << "pivoted conversion done\n";

  const std::string& main_string = column->data;
  size_t data_size = 0;
  for (int i=0; i<column->size; i++) data_size+=column->sizes[i];

  std::cout << "Total rows: " <<  column->size << "\n";

  int mp_t[256*256];
  memset(&mp_t, 0, sizeof(int)*256*256);
  std::map<std::string, int> mp;
  size_t n = column->size;
  auto bm_size = std::ceil((float)n / 64.);
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < n; i++)
  {
    auto off = column->offsets[i];
    for (int j = 1; j < column->sizes[i]; j++)
    {
      char prev = column->data[off + j - 1], curr = column->data[off + j];
      prev = to_lower(prev); curr = to_lower(curr);
      if (valid(prev) && valid(curr))
        mp_t[(prev)*256 + (curr)] = 1;
    }
  } // end of counting bitmasks
  // do a prefix sum to get the unique ids
  for (int i=1; i<256*256; i++) {
    mp_t[i] += mp_t[i-1];
  }
  auto stop = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
  std::cout << "Time to see all 2 char sequence: " << duration.count() << " ms\n";
  std::cout << "Possilbe 2 char sequence in text: " << mp_t[256*256 - 1] << "\n";
  std::cout << "Each bitmap is of size: " << bm_size * sizeof(uint64_t) / 1024 << " K bytes\n";

  start = std::chrono::high_resolution_clock::now();
  std::vector<std::vector<uint64_t>> bitmaps(mp_t[256*256 - 1] + 1, std::vector<uint64_t>(bm_size, 0));
  for (int i = 0; i < n; i++)
  {
    auto off = column->offsets[i];
    for (int j = 1; j < column->sizes[i]; j++)
    {
      char prev = column->data[off + j - 1], curr = column->data[off + j];
      prev = to_lower(prev); curr = to_lower(curr);
      if (valid(prev) && valid(curr))
        bitmaps[mp_t[(prev)*256 + (curr)]][i / 64] |= ((uint64_t)1 << (63 - (i % 64)));
    }
  }
  stop = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
  std::cout << "Time for building bitmap indices: " << duration.count() << " ms\n";


  std::vector<std::string> pats = {"%furiously special%", "%dependencies%", "%theodolites%", "%requests%", "%lyly%"};
  // std::vector<std::string> pats = {"%express%dependencies%", "%requests%ly%"};

  int* d_sizes, *d_matched_count, *d_sp_sizes;
  int64_t *d_offsets;
  int* d_prefix_table, *d_prefix_table_sizes;
  int64_t *d_wildcard_bm;
  char* d_data;
  char* d_pattern;
  cudaMalloc(&d_sizes, sizeof(int)*column->size);
  cudaMalloc(&d_offsets, sizeof(int64_t)*column->size);
  cudaMalloc(&d_data, sizeof(char)*data_size);
  cudaMalloc(&d_matched_count, sizeof(int));
  uint64_t *d_bitmask_res, *d_temp_bm;
  cudaMalloc(&d_bitmask_res, bm_size * sizeof(uint64_t));
  cudaMalloc(&d_temp_bm, bm_size * sizeof(uint64_t));
  int *d_prefilter_count;
  cudaMalloc(&d_prefilter_count, sizeof(int));
  cudaMemcpy(d_sizes, column->sizes, sizeof(int)*column->size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_offsets, column->offsets, sizeof(int64_t)*column->size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_data, column->data, sizeof(char)*data_size, cudaMemcpyHostToDevice);
  int64_t *d_filtered_indices;

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


  std::set<int> maps_to_consider;
  for (int i = 1; i < preprocessed_pattern.pattern.size(); i++)
  {
    char prev = preprocessed_pattern.pattern[i - 1], curr = preprocessed_pattern.pattern[i];
    prev = to_lower(prev); curr = to_lower(curr);
    int key = prev * 256 + curr; // ascii domain has 256 characters.
    maps_to_consider.insert(mp_t[key]);
  }
  // for (auto e: maps_to_consider) std::cout << e << " ";
  int TB = 256;
  cudaMemset(d_bitmask_res, 0xFF, bm_size * sizeof(uint64_t));
  for (auto e : maps_to_consider)
  {
    cudaMemcpy(d_temp_bm, bitmaps[e].data(), bm_size * sizeof(uint64_t), cudaMemcpyHostToDevice);
    gpu_and<<<std::ceil((float)bm_size / (float)TB), TB>>>(d_temp_bm, d_bitmask_res, bm_size);
  }
  int prefiltered_count;
  cudaMemset(d_prefilter_count, 0, sizeof(int));
  gpu_filter_count<<<std::ceil((float)bm_size / (float)TB), TB>>>(d_bitmask_res, d_prefilter_count, bm_size);
  cudaMemcpy(&prefiltered_count, d_prefilter_count, sizeof(int), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  CUDACHKERR();
  std::cout << "index selectivity: " << (float)prefiltered_count / (float)n << " prefilter count: " << prefiltered_count << "\n";
  cudaFree(d_filtered_indices);
  cudaMalloc(&d_filtered_indices, sizeof(int64_t) * prefiltered_count);
  cudaMemset(d_prefilter_count, 0, sizeof(int));
  gpu_gather_indices<<<std::ceil((float)bm_size / (float)TB), TB>>>(d_filtered_indices, d_bitmask_res, bm_size, d_prefilter_count);
  // try sorting the indices
  // std::vector<int> v(prefiltered_count);
  // cudaMemcpy(v.data(), d_filtered_indices, sizeof(int) * prefiltered_count, cudaMemcpyDeviceToHost);
  // sort(v.begin(), v.end());
  // cudaMemcpy(d_filtered_indices, v.data(), sizeof(int) * prefiltered_count, cudaMemcpyHostToDevice);


    // std::vector<int> TBs = {256};
    // for (auto TB: TBs) {
      std::cout << "Using thread block size: " << TB << std::endl;
    int gpu_matched_rows = 0;

    cudaMemset(d_matched_count, 0, sizeof(int));
    gpu_kmp_step<<<std::ceil((float)column->size/(float)TB), TB>>>(
      d_data, 
      d_offsets, 
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
    std::cout << "KMP STEP: " << gpu_matched_rows << "\n";
    

    std::vector<int> Ts = {32, 256};
    for (auto T: Ts) {

      cudaMemset(d_matched_count, 0, sizeof(int));
      gpu_kmp_step_pindex<<<std::ceil((float)column->size/(float)T), T>>>(
        d_data, 
        d_offsets, 
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
        d_prefix_table_sizes,
        d_bitmask_res,
        d_filtered_indices, 
        prefiltered_count
      );
      CUDACHKERR();
      cudaMemcpy(&gpu_matched_rows, d_matched_count, sizeof(int), cudaMemcpyDeviceToHost);
      CUDACHKERR();
      std::cout << "KMP STEP with PINDEX: " << gpu_matched_rows  << " TB: " << T << "\n";
    }
  std::cout << "actual selectivity: " << (float)gpu_matched_rows / (float)n << "\n";
    free(pattern);
  }
}