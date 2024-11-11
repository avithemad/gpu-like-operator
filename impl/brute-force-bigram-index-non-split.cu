#include "data.hpp"
#include <iostream>
#include <cassert>
#include "cudautils.cuh"
#include <vector>
#include <set>
#include <map>
#include <algorithm>
#include <chrono>
#include <omp.h>

__global__ void gpu_brute_force_prefilter(char *data, int *offsets, int *sizes, size_t table_size,
                                          char *pattern, int p_size, int *matched_count,
                                          uint64_t *bm,
                                          int *filtered_indices,
                                          int pf_count)
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
                                char *pattern, int p_size, int *matched_count)
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

__global__ void gpu_gather_indices(int *filtered_indices, uint64_t *bm_res, int size, int *count)
{
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid >= size)
    return;
  uint64_t temp = bm_res[tid];
  int base = tid * 64;
  for (int i = 0; i < 64; i++)
  {
    if (temp & 0x8000000000000000)
    {
      filtered_indices[atomicAdd(count, 1)] = base + i;
    }
    temp <<= 1;
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
bool valid(char c)
{
  return (c >= 'a' && c <= 'z');
}
char to_lower(char c)
{
  if ((c >= 'A' && c <= 'Z'))
    return (c - 'A') + 'a';
  else
  {
    return c;
  } 
}
struct ThreadArgs
{
  gpulike::StringColumn *comments_column;
  size_t start_idx;
  size_t end_idx;
  int res;
  std::string pattern;
};

void *process_comments(void *arg)
{
  ThreadArgs *args = (ThreadArgs *)arg;
  gpulike::StringColumn *comments_column = args->comments_column;
  size_t start = args->start_idx;
  size_t end = args->end_idx;

  // Process the elements from start to end (exclusive)
  auto pattern = args->pattern;
  for (size_t i = start; i < end; ++i)
  {
    for (int j = 0; j < (comments_column->sizes[i] - pattern.size() + 1); j++)
    {
      bool matched = true;
      for (int k = 0; k < (pattern.size()); k++)
      {
        if (comments_column->data[comments_column->offsets[i] + j + k] != pattern[k])
        {
          matched = false;
          break;
        }
      }
      if (matched)
      {
        args->res++;
        break;
      }
    }
  }

  return nullptr;
}
int cpu_brute_force(gpulike::StringColumn *comments_column, std::string pattern)
{
  int matched_rows = 0;
  // cpu side matching
  // # pragma omp parallel for reduction(+ : sum)
  // int sum = 0;
  //   for (int i = 0; i <= comments_column->size; ++i) {
  //       sum += i;
  // }
  const int num_threads = 12;
  pthread_t threads[num_threads];
  ThreadArgs args[num_threads];

  // Divide the work among 6 threads
  size_t chunk_size = comments_column->size / num_threads;
  size_t remainder = comments_column->size % num_threads;

  for (int i = 0; i < num_threads; ++i)
  {
    // Calculate start and end indices for each thread
    args[i].comments_column = comments_column;
    args[i].start_idx = i * chunk_size;
    args[i].end_idx = (i == num_threads - 1) ? comments_column->size : (i + 1) * chunk_size;
    args[i].res = 0;
    args[i].pattern = pattern;
    // Create each thread
    int ret = pthread_create(&threads[i], nullptr, process_comments, (void *)&args[i]);
    if (ret != 0)
    {
      std::cerr << "Error creating thread " << i << std::endl;
    }
  }

  // Join all threads
  for (int i = 0; i < num_threads; ++i)
  {
    pthread_join(threads[i], nullptr);
  }
  for (int i = 0; i < num_threads; i++)
  {
    matched_rows += args[i].res;
  }
  return matched_rows;
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

  int mp_t[256*256];
  memset(&mp_t, 0, sizeof(int)*256*256);
  std::map<std::string, int> mp;

  const char *pattern = argv[2];
  int p_size = ((std::string)pattern).size();
  auto bm_size = std::ceil((float)n / 64.);

  std::cout << "Executing brute force on CPU:\n";

  auto start = std::chrono::high_resolution_clock::now();
  int cpu_matched;
  // cpu_matched = cpu_brute_force(comments_column, pattern);
  auto stop = std::chrono::high_resolution_clock::now();
  std::cout << "Matches in cpu: " << cpu_matched << "\n";
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
  std::cout << "Time taken in cpu: " << duration.count() << " ms\n";

  start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < n; i++)
  {
    auto off = comments_column->offsets[i];
    for (int j = 1; j < comments_column->sizes[i]; j++)
    {
      char prev = comments_column->data[off + j - 1], curr = comments_column->data[off + j];
      prev = to_lower(prev); curr = to_lower(curr);
      if (valid(prev) && valid(curr))
        mp_t[(prev)*256 + (curr)] = 1;
    }
  }
  // do a prefix sum to get the unique ids
  for (int i=1; i<256*256; i++) {
    mp_t[i] += mp_t[i-1];
  }
  stop = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
  std::cout << "Time to see all 2 char sequence: " << duration.count() << " ms\n";
  std::cout << "Possilbe 2 char sequence in text: " << mp_t[256*256 - 1] << "\n";
  std::cout << "Each bitmap is of size: " << bm_size * sizeof(uint64_t) / 1024 << " K bytes\n";

  start = std::chrono::high_resolution_clock::now();
  std::vector<std::vector<uint64_t>> bitmaps(mp_t[256*256 - 1] + 1, std::vector<uint64_t>(bm_size, 0));
  for (int i = 0; i < n; i++)
  {
    auto off = comments_column->offsets[i];
    for (int j = 1; j < comments_column->sizes[i]; j++)
    {
      char prev = comments_column->data[off + j - 1], curr = comments_column->data[off + j];
      prev = to_lower(prev); curr = to_lower(curr);
      if (valid(prev) && valid(curr))
        bitmaps[mp_t[(prev)*256 + (curr)]][i / 64] |= ((uint64_t)1 << (63 - (i % 64)));
    }
  }
  stop = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
  std::cout << "Time for building bitmap indices: " << duration.count() << " ms\n";

  // simple check for a pattern of breach
  std::set<int> maps_to_consider;
  for (int i = 1; i < p_size; i++)
  {
    char prev = pattern[i - 1], curr = pattern[i];
    prev = to_lower(prev); curr = to_lower(curr);
    int key = prev * 256 + curr; // ascii domain has 256 characters.
    maps_to_consider.insert(mp_t[key]);
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
  std::cout << "index selectivity: " << (float)prefiltered_count / (float)n << " prefilter count: " << prefiltered_count << "\n";

  // gathering indices to filter in a separate array
  int *d_filtered_indices;
  cudaMemset(d_prefilter_count, 0, sizeof(int));
  cudaMalloc(&d_filtered_indices, sizeof(int) * prefiltered_count);
  gpu_gather_indices<<<std::ceil((float)bm_size / (float)TB), TB>>>(d_filtered_indices, d_bitmask_res, bm_size, d_prefilter_count);
  // try sorting the indices
  std::vector<int> v(prefiltered_count);
  cudaMemcpy(v.data(), d_filtered_indices, sizeof(int) * prefiltered_count, cudaMemcpyDeviceToHost);
  sort(v.begin(), v.end());
  cudaMemcpy(d_filtered_indices, v.data(), sizeof(int) * prefiltered_count, cudaMemcpyHostToDevice);

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