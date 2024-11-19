#pragma once

#include "data.hpp"
#include "pattern_utils.hpp"
#include "cudautils.cuh"

/**
 * Assumptions on the pattern:
 * 1. no duplicate %, if exists can be preprocessed
 */
__global__ void gpu_brute_force(
    char* data, 
    int* offsets,
    int* sizes, 
    size_t table_size, 
    int* matched_count, 
    char* pattern,
    int p_size, 
    int* p_subsizes,
    int p_subsizes_len,
    int percent_count, 
    uint64_t* bitmasks1d
) {  
  int tid = threadIdx.x + blockDim.x*blockIdx.x;
  if (tid >= table_size || p_size == 0) return;
  if (p_size == 1 && pattern[0] == '%') {
    atomicAdd(matched_count, 1);
    return;
  } 

  int offset = offsets[tid];
  int d_idx = 0;
  
  // substring search for the each subpattern
  for(int sp_idx = 0; sp_idx < p_subsizes_len; sp_idx++) {
    bool sp_match = false;
    int p_off = sp_idx > 0 ? p_subsizes[sp_idx-1] : 0;
    for (int i = d_idx; i < sizes[tid] - p_subsizes[sp_idx] + 1; i++) {
        bool matched = true;
        for (int j = 0; j < p_subsizes[sp_idx]; j++) {
            if (data[offset + i + j] != pattern[p_off + j]) {
                matched = false; break;
            }
        }
        if (matched) {
            sp_match = true;
            d_idx = i + p_subsizes[sp_idx]; // match occured at i, now start the next search from this idx + subpattern_len
            break;
        }
    }
    if (!sp_match) {
        return; // one of the subpattern did not match so return
    }
  }
  // execution reaches here if and only if all subpatterns are matched
  atomicAdd(matched_count, 1);
}