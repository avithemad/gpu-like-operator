#include "data.hpp"
#include "pattern_utils.hpp"
#include <iostream>
#include <cassert>
#include <vector>
#include <bitset>
#include "cudautils.cuh"

__global__ void gpu_brute_force(char* data, int* offsets, int* sizes, size_t table_size, char* pattern, int p_size, int* matched_count) {
  int tid = threadIdx.x + blockDim.x*blockIdx.x;
  if (tid >= table_size) return;
  // printf("GPU:%s\n", pattern);
  bool done = false;
  for (int str_index=0; str_index<sizes[tid] - p_size + 1; str_index++) {
    bool matched = true;
    for (int pattern_index=0; pattern_index<p_size; pattern_index++) {
      if (data[offsets[tid] + pattern_index + str_index] != pattern[pattern_index]) {
       matched = false;
      //  break; 
      }
    }
    if (matched) {
      if (!done)
      atomicAdd(matched_count, 1);
      done = true;
    }
  }
}
__global__ void gpu_brute_force_limited(char* data, int* offsets, int* sizes, size_t table_size, char* pattern, int p_size, int* matched_count) {
  int tid = threadIdx.x + blockDim.x*blockIdx.x;
  if (tid >= table_size) return;
  for (int str_index=0; str_index<sizes[tid] - p_size + 1; str_index++) {
    bool matched = true;
    for (int pattern_index=0; pattern_index<p_size; pattern_index++) {
      if (data[offsets[tid] + pattern_index + str_index] != pattern[pattern_index]) {
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


__global__ void gpu_brute_force_Purr(
    char* data, int* offsets,
    int* sizes, size_t table_size, 
    int* matched_count, char* pattern,
    int p_size, int per_count, 
    uint64_t* bitmasks1d) {

    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid >= table_size) return;

    // Precompute base address for the current thread's string
    char* string_data = data + offsets[tid];
    int string_size = sizes[tid];

    // Check if the first character is '%' or not
    bool starts_with_percent = (pattern[0] == '%');

    // If pattern does not start with '%', perform a prefix match only
    if (!starts_with_percent) {
        bool matched = true;

        for (int pattern_index = 0; pattern_index < p_size; pattern_index++) {
            char current_pattern = pattern[pattern_index];

            // Handle character ranges ([]) using bitmasks
            if (current_pattern == '[') {
                int char_value = string_data[pattern_index];
                int block_index = char_value / BITS_PER_BLOCK;
                int position_within_block = char_value % BITS_PER_BLOCK;

                if (!(bitmasks1d[block_index] >> position_within_block & 1)) {
                    matched = false;
                    break;
                }
            } else if (current_pattern != '_' && string_data[pattern_index] != current_pattern) {
                matched = false;
                break;
            }
        }

        if (matched) {
            atomicAdd(matched_count, 1);
        }
        return; // No need for further checks
    }

    // Standard matching for patterns starting with '%'
    int max_start = string_size - p_size + per_count + 1;
    for (int str_index = 0; str_index < max_start; str_index++) {
        bool matched = true;
        int pattern_offset = 0;
        int mask_index = 0;

        for (int pattern_index = 0; pattern_index < p_size; pattern_index++) {
            char current_pattern = pattern[pattern_index];

            // Handle '%' wildcard by adjusting pattern_offset
            if (current_pattern == '%') {
                pattern_offset++;
                continue;
            }

            // Handle character ranges ([]) using bitmasks
            if (current_pattern == '[') {
                int char_value = string_data[str_index + pattern_index - pattern_offset];
                int block_index = char_value / BITS_PER_BLOCK;
                int position_within_block = char_value % BITS_PER_BLOCK;

                if (!(bitmasks1d[mask_index * 4 + block_index] >> position_within_block & 1)) {
                    matched = false;
                    break;
                }
                mask_index++;
                continue;
            }

            // Direct character match (excluding '_')
            if (current_pattern != '_' && 
                string_data[str_index + pattern_index - pattern_offset] != current_pattern) {
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

__global__ void gpu_brute_force_Purr_shared(
    char* data, int* offsets,
    int* sizes, size_t table_size, 
    int* matched_count, char* pattern,
    int p_size, int per_count, 
    uint64_t* bitmasks1d) {

    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid >= table_size) return;

    // Load the size of the current string
    int string_size = sizes[tid];
    char* string_data = data + offsets[tid];

    // Check if the first character is '%' or not
    bool starts_with_percent = (pattern[0] == '%');

    // Shared memory optimization for pattern
    extern __shared__ char shared_pattern[];
    if (threadIdx.x < p_size) {
        shared_pattern[threadIdx.x] = pattern[threadIdx.x];
    }
    __syncthreads();

    // If pattern does not start with '%', perform a prefix match only
    if (!starts_with_percent) {
        bool matched = true;

        for (int pattern_index = 0; pattern_index < p_size; pattern_index++) {
            char current_pattern = shared_pattern[pattern_index];

            // Handle character ranges ([]) using bitmasks
            if (current_pattern == '[') {
                int char_value = string_data[pattern_index];
                int block_index = char_value / BITS_PER_BLOCK;
                int position_within_block = char_value % BITS_PER_BLOCK;

                if (!(bitmasks1d[block_index] >> position_within_block & 1)) {
                    matched = false;
                    break;
                }
            } else if (current_pattern != '_' && string_data[pattern_index] != current_pattern) {
                matched = false;
                break;
            }
        }

        if (matched) {
            atomicAdd(matched_count, 1);
        }
        return; // No need for further checks
    }

    // Standard matching for patterns starting with '%'
    int max_start = string_size - p_size + per_count + 1;
    for (int str_index = 0; str_index < max_start; str_index++) {
        bool matched = true;
        int pattern_offset = 0;
        int mask_index = 0;

        for (int pattern_index = 0; pattern_index < p_size; pattern_index++) {
            char current_pattern = shared_pattern[pattern_index];

            // Handle '%' wildcard by adjusting pattern_offset
            if (current_pattern == '%') {
                pattern_offset++;
                continue;
            }

            // Handle character ranges ([]) using bitmasks
            if (current_pattern == '[') {
                int char_value = string_data[str_index + pattern_index - pattern_offset];
                int block_index = char_value / BITS_PER_BLOCK;
                int position_within_block = char_value % BITS_PER_BLOCK;

                if (!(bitmasks1d[mask_index * 4 + block_index] >> position_within_block & 1)) {
                    matched = false;
                    break;
                }
                mask_index++;
                continue;
            }

            // Direct character match (excluding '_')
            if (current_pattern != '_' && 
                string_data[str_index + pattern_index - pattern_offset] != current_pattern) {
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



int cpu_brute_force_noVec(
    gpulike::StringColumn* comments_column, 
    const std::string& pattern, 
    int p_size, 
    int per_count, 
    const std::vector<std::vector<uint64_t>>& bitmasks) {
    
    int matched_rows = 0;
    int match_start_index, pattern_offset, mask_index;

    if (pattern[0]!='%') {
        for (int i = 0; i < comments_column->size; i++) {
            // Check only the first substring of length p_size
            bool prefix_matched = true;
            for (int j = 0; j < p_size; j++) {
                if (pattern[j] != '_' && 
                    comments_column->data[comments_column->offsets[i] + j] != pattern[j]) {
                    prefix_matched = false;
                    break;
                }
            }
            if (prefix_matched) {
                matched_rows++;
            }
        }
        return matched_rows; // No need for further matching
    }
    for (int i = 0; i < comments_column->size; i++) {
        match_start_index = 0; pattern_offset = 0;

        // Loop through possible starting positions in the current string
        for (int str_index = 0; str_index < (comments_column->sizes[i] - p_size + per_count + 1); str_index++) {
            bool matched = true;
            mask_index = 0;

            // Loop through pattern characters for matching
            for (int pattern_index = match_start_index; pattern_index < p_size; pattern_index++) {
                char current_pattern = pattern[pattern_index];

                // Handle '%' wildcard by adjusting match start index and pattern offset
                if (current_pattern == '%') {
                    match_start_index = pattern_index + 1;
                    pattern_offset++;
                    continue;
                }

                // Handle character ranges ([]) using bitmasks
                if (current_pattern == '[') {
                    int data_index = comments_column->data[comments_column->offsets[i] + str_index + pattern_index - pattern_offset];
                    int block_index = data_index / BITS_PER_BLOCK;
                    int position_within_block = data_index % BITS_PER_BLOCK;

                    if (!(bitmasks[mask_index][block_index] >> position_within_block & 1)) {
                        matched = false;
                        break;
                    }
                    mask_index++;
                    continue;
                }

                // Direct character match (excluding '_')
                if (current_pattern != '_' && 
                    comments_column->data[comments_column->offsets[i] + str_index + pattern_index - pattern_offset] != current_pattern) {
                    matched = false;
                    break;
                }
            }

            // If pattern is fully matched, increment matched_rows and move to next row
            if (matched) {
                matched_rows++;
                break;
            }
        }
    }

    return matched_rows;
}



int cpu_brute_force(gpulike::StringColumn* comments_column, std::string pattern) {
  int matched_rows = 0;
  std::vector<std::string> patterns;
  patterns=splitByPercentage(pattern);
  // keeping count of block_indexber of patterns matched
  int match_start_index=0;
  // cpu side matching
  for (int i=0; i<comments_column->size; i++) {
    match_start_index=0;
    for (int str_index=0; str_index<(comments_column->sizes[i] - patterns[match_start_index].size() + 1); str_index++) {
      // matching done here
      bool matched = true;
      for (int pattern_index=0; pattern_index<(patterns[match_start_index].size()); pattern_index++) {
        if (pattern[pattern_index]!='_' && comments_column->data[comments_column->offsets[i]+str_index+pattern_index]!=patterns[match_start_index][pattern_index]) 
        {
          matched = false;
          break;  
        }

      }
      if (matched) {
        match_start_index++;
        if(match_start_index==patterns.size()){
          matched_rows++;
          break;
        }
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

  char* pattern = argv[2];
  int p_size = ((std::string)pattern).size(); 
  int per_count=count_per(pattern);
  // ok cuda doesnt suppoer bitset so use uint
  std::vector<std::vector<uint64_t>> bitmasks=createBitmasks(pattern,p_size);
  int cpu_matched_rows;
  
  // cpu_matched_rows = cpu_brute_force(comments_column, pattern);
  // std::cout << "Total matched rows in CPU: " << cpu_matched_rows << "\n";
  // std::vector<std::bitset<128>> bitmasks=createBitmasks(pattern, p_size);
  cpu_matched_rows = cpu_brute_force_noVec(comments_column, pattern, p_size, per_count, bitmasks);



  std::cout << "Total matched rows in CPU: " << cpu_matched_rows << "\n";

  std::cout << "Now brute forcing in GPU\n"; 
  int* d_sizes, *d_matched_count, *d_matched_count_2,*d_matched_count_3,*d_matched_count_4;
  int* d_offsets;
  char* d_data;
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
  cudaMalloc(&d_matched_count_2, sizeof(int));
  cudaMalloc(&d_matched_count_3, sizeof(int));
   cudaMalloc(&d_matched_count_4, sizeof(int));
  cudaMalloc(&d_offsets, sizeof(int)*comments_column->size);
  cudaMalloc(&d_data, sizeof(char)*data_size);

  cudaMemcpy(bitmasks1d, h_flattened.data(), total_elements * sizeof(uint64_t), cudaMemcpyHostToDevice);
  cudaMemcpy(d_sizes, comments_column->sizes, sizeof(int)*comments_column->size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_offsets, comments_column->offsets, sizeof(int)*comments_column->size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_data, comments_column->data, sizeof(char)*data_size, cudaMemcpyHostToDevice);

  cudaMemset(d_matched_count, 0, sizeof(int));
  cudaMemset(d_matched_count_2, 0, sizeof(int));
  cudaMemset(d_matched_count_3, 0, sizeof(int));
   cudaMemset(d_matched_count_4, 0, sizeof(int));
  CUDACHKERR();

  int TB = 256;
  char* d_pattern;
  int gpu_matched_rows = 0;
  cudaMalloc(&d_pattern, sizeof(char)*p_size);
  cudaMemcpy(d_pattern, pattern, sizeof(char)*p_size, cudaMemcpyHostToDevice);
  gpu_brute_force<<<std::ceil((float)comments_column->size/(float)TB), TB>>>(d_data, d_offsets, d_sizes, comments_column->size, d_pattern, p_size, d_matched_count);
  // gpu_brute_force_limited<<<std::ceil((float)comments_column->size/(float)TB), TB>>>(d_data, d_offsets, d_sizes, comments_column->size, d_pattern, p_size, d_matched_count_2);
  gpu_brute_force_Purr<<<std::ceil((float)comments_column->size/(float)TB), TB>>>(d_data, d_offsets, d_sizes, comments_column->size, d_matched_count_3,d_pattern, p_size, per_count, bitmasks1d);
   gpu_brute_force_Purr_shared<<<std::ceil((float)comments_column->size/(float)TB), TB>>>(d_data, d_offsets, d_sizes, comments_column->size, d_matched_count_4,d_pattern, p_size, per_count, bitmasks1d);
  
  CUDACHKERR();
  cudaMemcpy(&gpu_matched_rows, d_matched_count, sizeof(int), cudaMemcpyDeviceToHost);
  std::cout << "Result from GPU: " << gpu_matched_rows << "\n";
  cudaMemcpy(&gpu_matched_rows, d_matched_count_3, sizeof(int), cudaMemcpyDeviceToHost);
  std::cout << "Result from GPU_PURR: " << gpu_matched_rows << "\n";
  CUDACHKERR();
  cudaMemcpy(&gpu_matched_rows, d_matched_count_4, sizeof(int), cudaMemcpyDeviceToHost); 
    std::cout << "Result from GPU_PURR_shared: " << gpu_matched_rows << "\n";
  CUDACHKERR();
  // assert(gpu_matched_rows == cpu_matched_rows);

}
// ab,311001
// s_b,196162
// _,6001216
// a__le,654335
// [a-match_start_index],6001112
// b[a-m]_%a,432803