#include "data.hpp"
#include <iostream>
#include <cassert>
#include "cudautils.cuh"
#include <vector>
#include <set>
#include <map>
#include <algorithm>

__global__ void gpu_brute_force_prefilter(char *data, int *offsets, int *sizes, size_t table_size,
                                          char *pattern, int p_size, int *matched_count,
                                          uint64_t *bm,
                                          int* filtered_indices,
                                          int pf_count
                                          )
{
  int c_tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (c_tid >= pf_count)
    return;

  // prefilter based on bigram
  int tid = filtered_indices[c_tid];
  // bool m = bm[tid/64] & ((uint64_t)1 << (63 - (tid%64)));
  // if (!m)return;

  for (int j = 0; j < sizes[tid] - p_size + 1; j++)
  {
    bool matched = true;
    for (int k = 0; k < p_size; k++)
    {
      if (data[offsets[tid] + k + j] != pattern[k])
      {
        matched = false;
        break;
      }
    }
    if (matched)
    {
      atomicAdd(matched_count, 1);
      break;
    }
  }
}
__global__ void gpu_brute_force(char *data, int *offsets, int *sizes, size_t table_size, 
char *pattern, int p_size, int *matched_count
)
{
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid >= table_size)
    return;

  // prefilter based on bigram

  for (int j = 0; j < sizes[tid] - p_size + 1; j++)
  {
    bool matched = true;
    for (int k = 0; k < p_size; k++)
    {
      if (data[offsets[tid] + k + j] != pattern[k])
      {
        matched = false;
        break;
      }
    }
    if (matched)
    {
      atomicAdd(matched_count, 1);
      break;
    }
  }
}

__global__ void gpu_gather_indices(int* filtered_indices, uint64_t *bm_res, int size, int* count) {
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid >= size)
    return;
  uint64_t temp = bm_res[tid];
  int base = tid * 64;
  for (int i = 0; i < 64; i++)
  {
    if (temp & 0x8000000000000000) {
      filtered_indices[atomicAdd(count, 1)] = base + i;
    }
    temp <<=1;
  }
}

__global__ void gpu_filter_count(uint64_t *bm, int *res, int size)
{
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid >= size)
    return;
  int r = 0;
  uint64_t temp = bm[tid];
  for (int i = 0; i < 64; i++)
  {
    r += ((temp & 0x8000000000000000) != 0);
    temp <<= 1;
  }
  atomicAdd(res, r);
}

__global__ void gpu_and(uint64_t *temp, uint64_t *res, int size)
{
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid >= size)
    return;
  res[tid] &= temp[tid];
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
  const std::string &main_string = comments_column->data;
  size_t data_size = 0;
  for (int i = 0; i < comments_column->size; i++)
    data_size += comments_column->sizes[i];

  std::cout << "Total rows: " << comments_column->size << "\n";
  size_t n = comments_column->size;

  // std::map<std::string, int> mp;
  // for (int i=0; i<comments_column->size; i++) {
  //   int off = comments_column->offsets[i];
  //   for (int j=1; j<comments_column->sizes[i]; j++) {
  //     std::string a = "";
  //     a.push_back(comments_column->data[off + j - 1]);
  //     a.push_back(comments_column->data[off + j]);
  //     mp[a]++;
  //   }
  // }
  // std::cout << "Possilbe 2 char sequ: " << mp.size() << "\n";
  // for (auto p: mp) {
  //   std::cout << p.first << " " << p.second << "\n";
  // }

  const char *pattern = argv[2];
  int p_size = ((std::string)pattern).size();
  auto bm_size = std::ceil((float)n / 64.);

  std::cout << "Building bitmap index of the whole ascii set of size: " << bm_size << "\n";

  // construct 50 bitmaps
  std::vector<std::vector<uint64_t>> bitmaps(512, std::vector<uint64_t>(bm_size, 0));
  for (int i = 0; i < n; i++)
  {
    auto off = comments_column->offsets[i];
    for (int j = 1; j < comments_column->sizes[i]; j++)
    {
      char prev = comments_column->data[off + j - 1], curr = comments_column->data[off + j];
      // decompose this key into base <10,10,10,10,10> index
      bitmaps[prev][i/64] |= ((uint64_t)1 << (63 - (i % 64)));
      bitmaps[curr + 256][i/64] |= ((uint64_t)1 << (63 - (i % 64))); 
      // if (prev == 'b' && curr == 'r') {
      //   printf("br: %lx\n", bitmaps[prev][0]);
      // }
    }
  }

  // simple check for a pattern of breach
  std::set<int> maps_to_consider;
  for (int i = 1; i < p_size; i++)
  {
    char prev = pattern[i - 1], curr = pattern[i];
    // int key = prev * 256 + curr; // ascii domain has 256 characters.
    maps_to_consider.insert(prev);
    maps_to_consider.insert(curr + 256);
  }
  // for (auto e: maps_to_consider) std::cout << e << " ";
  uint64_t *d_bitmask_res, *d_temp_bm;
  int TB = 256;
  cudaMalloc(&d_bitmask_res, bm_size * sizeof(uint64_t));
  cudaMalloc(&d_temp_bm, bm_size * sizeof(uint64_t));
  cudaMemset(d_bitmask_res, 0xFFFFFFFFFFFFFFFF, bm_size * sizeof(uint64_t));
  for (auto e : maps_to_consider)
  {
    cudaMemcpy(d_temp_bm, bitmaps[e].data(), bm_size * sizeof(uint64_t), cudaMemcpyHostToDevice);
    gpu_and<<<std::ceil((float)bm_size / (float)TB), TB>>>(d_temp_bm, d_bitmask_res, bm_size);
  }
  int prefiltered_count, *d_prefilter_count;
  cudaMalloc(&d_prefilter_count, sizeof(int));
  cudaMemset(d_prefilter_count, 0, sizeof(int));
  gpu_filter_count<<<std::ceil((float)bm_size / (float)TB), TB>>>(d_bitmask_res, d_prefilter_count, bm_size);
  cudaMemcpy(&prefiltered_count, d_prefilter_count, sizeof(int), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  CUDACHKERR();
  std::cout << "index selectivity: " << (float)prefiltered_count / (float)n  << " prefilter count: " << prefiltered_count << "\n";

  uint64_t x;
  cudaMemcpy(&x, d_bitmask_res, sizeof(uint64_t), cudaMemcpyDeviceToHost);
  printf("%lx\n", x);

  // gathering indices to filter in a separate array
  int *d_filtered_indices;
  cudaMemset(d_prefilter_count, 0, sizeof(int));
  cudaMalloc(&d_filtered_indices, sizeof(int)*prefiltered_count);
  gpu_gather_indices<<<std::ceil((float)bm_size / (float)TB), TB>>>(d_filtered_indices, d_bitmask_res, bm_size, d_prefilter_count);
  // try sorting the indices
  std::vector<int> v(prefiltered_count);
  cudaMemcpy(v.data(), d_filtered_indices, sizeof(int)*prefiltered_count, cudaMemcpyDeviceToHost);
  sort(v.begin(), v.end());
  cudaMemcpy(d_filtered_indices, v.data(), sizeof(int)*prefiltered_count, cudaMemcpyHostToDevice);


  std::cout << "Now brute forcing in GPU\n";
  int *d_sizes, *d_matched_count;
  int *d_offsets;
  char *d_data;
  cudaMalloc(&d_sizes, sizeof(int) * comments_column->size);
  cudaMalloc(&d_matched_count, sizeof(int));
  cudaMalloc(&d_offsets, sizeof(int) * comments_column->size);
  cudaMalloc(&d_data, sizeof(char) * data_size);

  cudaMemcpy(d_sizes, comments_column->sizes, sizeof(int) * comments_column->size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_offsets, comments_column->offsets, sizeof(int) * comments_column->size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_data, comments_column->data, sizeof(char) * data_size, cudaMemcpyHostToDevice);
  cudaMemset(d_matched_count, 0, sizeof(int));
  CUDACHKERR();

  char *d_pattern;
  cudaMalloc(&d_pattern, sizeof(char) * p_size);
  cudaMemcpy(d_pattern, pattern, sizeof(char) * p_size, cudaMemcpyHostToDevice);
  gpu_brute_force_prefilter<<<std::ceil((float)prefiltered_count / (float)32), 32>>>(
      d_data, d_offsets, d_sizes, comments_column->size, d_pattern, p_size, d_matched_count, d_bitmask_res,
      d_filtered_indices, prefiltered_count);
  CUDACHKERR();
  int gpu_matched_rows_bm = 0;
  cudaMemcpy(&gpu_matched_rows_bm, d_matched_count, sizeof(int), cudaMemcpyDeviceToHost);

  cudaMemset(d_matched_count, 0, sizeof(int));
  gpu_brute_force<<<std::ceil((float)n / (float)TB), TB>>>(
      d_data, d_offsets, d_sizes, comments_column->size, d_pattern, p_size, d_matched_count);
  int gpu_matched_rows_bf = 0;
  cudaMemcpy(&gpu_matched_rows_bf, d_matched_count, sizeof(int), cudaMemcpyDeviceToHost);
  CUDACHKERR();
  // assert(gpu_matched_rows_bf == gpu_matched_rows_bm);

  std::cout << "Result from brute force prefilter: " << gpu_matched_rows_bm << "\n";
  std::cout << "Result from brute force : " << gpu_matched_rows_bf << "\n";
  std::cout << "actual selectivity: " << (float)gpu_matched_rows_bm / (float)n << "\n";
}