#include "data.hpp"
#include <iostream>
#include <cassert>
#include "cudautils.cuh"
#include <vector>
#include <set>
#include <map>
#include <algorithm>
#include <chrono>

__global__ void gpu_gather_indices(int64_t *filtered_indices, uint64_t *bm_res, int size, int *count)
{
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid >= size)
    return;
  uint64_t temp = bm_res[tid];
  int64_t base = tid * 64;
  for (int64_t i = 0; i < 64; i++)
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
  return true;
  return !(c== '|' || (c >= '0' && c <= '9'));
}

__global__ void gpu_kmp_step_pindex(
  char *data, 
  int64_t *offsets, 
  int *sizes, 
  size_t table_size,
  int *result,
  char *pattern, 
  int p_size, 
  int* sp_sizes, //  at minimum it will have 2 elements with size 0
  int sp_sizes_size,
  int64_t* bitmasks,
  int bitmasks_size,
  int* prefix_tables,
  int* prefix_table_sizes,
  uint64_t *bm,
  int64_t *filtered_indices,
  int pf_count
)
{
  int64_t c_tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (c_tid >= pf_count)
    return;

  // prefilter based on bigram
  int64_t tid = filtered_indices[c_tid];
  if (p_size > sizes[tid]) return; // trivial non match, pattern size greater than text size

  int64_t offset = offsets[tid];
  int s_start = sp_sizes[0];
  int s_end = sizes[tid] - sp_sizes[sp_sizes_size - 1]; // range where sequential substring search TBD
  // starts with
  // TODO: add bitmask check and wildcard check
  int bm_idx = 0;
  for (int64_t i=0; i<sp_sizes[0]; i++) {
    if (pattern[i] == '_') continue;
    if (pattern[i] == '[') {
      if ((bitmasks[bm_idx*4 + data[offset + i]/64] & ((int64_t)1 << (data[offset + i]%64))) == 0) return;
      bm_idx++;
      continue;
    }
    if (data[offset + i] != pattern[i]) return;
  }
  if (sp_sizes[1] == -1) { // exact match condition
    if (sp_sizes[0] == sizes[tid])
      atomicAdd(result, 1);
    return;
  } 
  int last_bm_idx = bitmasks_size - 1;
  // ends with
  for (int64_t i=0; i<sp_sizes[sp_sizes_size - 1]; i++) {
    if (pattern[p_size - 1 - i] == '_') continue;
    if (pattern[p_size - 1 - i] == '[') {
      if ((bitmasks[last_bm_idx*4 + data[offset + sizes[tid] - 1 - i]/64] & 
          ((int64_t)1 << (data[offset + sizes[tid] - 1 - i]%64))) == 0) return;
      last_bm_idx--;
      continue;
    }
    if (data[offset + sizes[tid] - 1 - i] != pattern[p_size - 1 - i]) return;
  }
  // up until here threads proceed in lock step  
  int p_offset = sp_sizes[0];
  int pref_table_offset = 0;
  for (int64_t i=1; i < sp_sizes_size-1; i++) { // for each subpattern, conduct substring matching
    bool sp_match = false;
    int64_t k = 0;
    for (int64_t j = s_start; j < s_end; j++) {
      // wait until other threads decide to move to next index
      while (k > 0 && pattern[p_offset + k]!=data[offset + j]) 
        k = prefix_tables[pref_table_offset + k - 1];
      k += (pattern[p_offset + k] == data[offset + j]);
      if (k == prefix_table_sizes[i-1]) {
        s_start = j + 1;
        sp_match = true;
        break;
      }
    }
    if (!sp_match) return; 

    p_offset += sp_sizes[i];
    pref_table_offset += prefix_table_sizes[i-1];
  }
  atomicAdd(result, 1);
}