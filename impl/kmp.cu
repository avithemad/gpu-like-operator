#include "data.hpp"
#include <iostream>
#include <cassert>
#include <vector>
#include "cudautils.cuh"
#include "pattern_utils.hpp"
#include <cstring> 



int cpu_kmp(gpulike::StringColumn* comments_column, std::string pattern, int* prefix) {
    // Check if the first character is '%' or not
    bool starts_with_percent = (pattern[0] == '%');
    int matched_rows = 0;

    // If the pattern does not start with '%', perform a prefix match only
    if (!starts_with_percent) {
        for (int i = 0; i < comments_column->size; i++) {
            int q = 0; // Keeps track of the prefix length

            // Loop through the first part of the string
            for (int j = 0; j < comments_column->sizes[i]; j++) {
                while (q > 0 && pattern[q] != comments_column->data[comments_column->offsets[i] + j]) {
                    q = prefix[q - 1];
                }
                if (pattern[q] == comments_column->data[comments_column->offsets[i] + j]) {
                    q++;
                }
                if (q == pattern.size()) {
                    matched_rows++;
                    break;
                }
                // If the prefix doesn't match at the start, stop early
                if (j == pattern.size() - 1 && q < pattern.size()) {
                    break;
                }
            }
        }
        return matched_rows; // No need for further checks
    }

    // Standard KMP-based matching for patterns with '%'
    int q = 0, base = 0;
    for (int i = 0; i < comments_column->size; i++) {
        q = 0;
        base = 0;
        for (int j = 0; j < comments_column->sizes[i]; j++) {
            // Handle '%' wildcard
            if (pattern[q] == '%') {
                q++;
                base = q; // Reset base to the new start after '%'
            }

            // Use prefix array to backtrack on mismatch
            while (q > base && pattern[q] != comments_column->data[comments_column->offsets[i] + j]) {
                q = prefix[q - 1];
            }

            // Advance match position if characters match
            if (pattern[q] == comments_column->data[comments_column->offsets[i] + j]) {
                q++;
            }

            // If full pattern matches, count and break
            if (q == pattern.size()) {
                matched_rows++;
                break;
            }
        }
    }

    return matched_rows;
}

__global__ void gpu_kmp(
    char* data, int* offsets, int* sizes, 
    size_t table_size, int* matched_count,
    int* prefix, char* pattern, int p_size) {

    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid >= table_size) return;

    int q = 0; // Keeps track of the matched prefix length
    int base = 0; // Tracks progress for subpatterns split by '%'

    // Check if the pattern starts with '%' or not
    bool starts_with_percent = (pattern[0] == '%');

    // If pattern does not start with '%', perform a prefix-only match
    if (!starts_with_percent) {
        for (int j = 0; j < sizes[tid]; j++) {
            while (q > 0 && pattern[q] != data[offsets[tid] + j]) {
                q = prefix[q - 1]; // Backtrack using prefix table
            }
            if (pattern[q] == data[offsets[tid] + j]) {
                q++;
            }
            if (q == p_size) {
                atomicAdd(matched_count, 1);
                return; // Exit as soon as a match is found
            }
            // If prefix doesn't match and we reach the end of the relevant substring, exit
            if (j == p_size - 1 && q < p_size) {
                return;
            }
        }
        return; // No match found, exit
    }

    // Standard KMP-based matching for patterns with '%'
    q = 0;
    for (int j = 0; j < sizes[tid]; j++) {
        // Handle '%' wildcard
        if (pattern[q] == '%') {
            q++;
            base = q; // Reset base to the new start after '%'
        }

        // Backtrack on mismatch
        while (q > base && pattern[q] != data[offsets[tid] + j]) {
            q = prefix[q - 1];
        }

        // Advance match position if characters match
        if (pattern[q] == data[offsets[tid] + j]) {
            q++;
        }

        // If full pattern matches, count and exit
        if (q == p_size) {
            atomicAdd(matched_count, 1);
            return; // Exit as soon as a match is found
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

    int match_start_index = 0, pattern_offset = 0, mask_index = 0;
    int block_index, position_within_block;

    // Check if the first character is '%' or not
    bool starts_with_percent = (pattern[0] == '%');

    // If pattern does not start with '%', perform a prefix match only
    if (!starts_with_percent) {
        bool matched = true;
        for (int pattern_index = 0; pattern_index < p_size; pattern_index++) {
            char current_pattern = pattern[pattern_index];

            // Handle character ranges ([]) using bitmasks
            if (current_pattern == '[') {
                block_index = data[offsets[tid] + pattern_index] / BITS_PER_BLOCK;
                position_within_block = data[offsets[tid] + pattern_index] % BITS_PER_BLOCK;

                if (!(bitmasks1d[mask_index * 4 + block_index] >> position_within_block & 1)) {
                    matched = false;
                    break;
                }
                mask_index++;
                continue;
            }

            // Direct character match (excluding '_')
            if (current_pattern != '_' && 
                data[offsets[tid] + pattern_index] != current_pattern) {
                matched = false;
                break;
            }
        }

        // If the prefix matches, count this row as a match
        if (matched) {
            atomicAdd(matched_count, 1);
        }
        return; // No need for further checks
    }

    // Standard matching when pattern starts with '%'
    for (int str_index = 0; str_index < sizes[tid] - p_size + per_count + 1; str_index++) {
        bool matched = true;
        mask_index = 0;

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
                block_index = data[offsets[tid] + str_index + pattern_index - pattern_offset] / BITS_PER_BLOCK;
                position_within_block = data[offsets[tid] + str_index + pattern_index - pattern_offset] % BITS_PER_BLOCK;

                if (!(bitmasks1d[mask_index * 4 + block_index] >> position_within_block & 1)) {
                    matched = false;
                    break;
                }
                mask_index++;
                continue;
            }

            // Direct character match (excluding '_')
            if (current_pattern != '_' && 
                data[offsets[tid] + str_index + pattern_index - pattern_offset] != current_pattern) {
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




// didnt use split subpatterns in vector

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



int main(int argc, char* argv[]) {

    if (argc < 3) {
        std::cout << "Usage: ./kmp <path_to_string_column_file> <pattern>\n";
        std::cout << "Example: ./kmp /media/db/comments.txt ab%b\n";
        return 1; // Exit with error
    }

    std::string txt_file = argv[1];
    std::string pattern = argv[2]; // Pattern from command line
    
    gpulike::StringColumn* comments_column = gpulike::read_txt(txt_file);
    if (comments_column == nullptr) {
        std::cout << "Unable to read comments column, possibly no data in the file.\n";
        return 1; // Exit with error
    }

    const std::string& main_string = comments_column->data;
    size_t data_size = 0;
    for (int i = 0; i < comments_column->size; i++) {
        data_size += comments_column->sizes[i];
    }

    std::cout << "Total rows: " << comments_column->size << "\n";
    int per_count=count_per(pattern);
    // Precompute KMP prefix array
    int* prefix = compute_prefix(pattern);

    // CPU KMP
    int cpu_matched_rows = cpu_kmp(comments_column, pattern, prefix);
    std::cout << "Total matched rows in CPU KMP: " << cpu_matched_rows << "\n";

    // CPU Brute Force
    std::vector<std::vector<uint64_t>> bitmasks ;
    cpu_matched_rows = cpu_brute_force_noVec(comments_column, pattern,pattern.size(),per_count, bitmasks);
    std::cout << "Total matched rows in CPU brute force: " << cpu_matched_rows << "\n";

    std::cout << "Now running KMP on GPU\n";

    // Allocate device memory
    int* d_sizes, *d_offsets, *d_matched_count, *d_prefix;
    char* d_data, *d_pattern;

    cudaMalloc(&d_sizes, sizeof(int) * comments_column->size);
    cudaMalloc(&d_offsets, sizeof(int) * comments_column->size);
    cudaMalloc(&d_matched_count, sizeof(int));
    cudaMalloc(&d_data, sizeof(char) * data_size);
    cudaMalloc(&d_pattern, sizeof(char) * (pattern.size() + 1)); // Include null terminator
    cudaMalloc(&d_prefix, sizeof(int) * pattern.size());

    // Copy data to device memory
    cudaMemcpy(d_sizes, comments_column->sizes, sizeof(int) * comments_column->size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_offsets, comments_column->offsets, sizeof(int) * comments_column->size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_data, comments_column->data, sizeof(char) * data_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_pattern, pattern.c_str(), sizeof(char) * (pattern.size() + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(d_prefix, prefix, sizeof(int) * pattern.size(), cudaMemcpyHostToDevice);

    cudaMemset(d_matched_count, 0, sizeof(int));
    CUDACHKERR();

    // Launch GPU kernel
    int TB = 32; // Threads per block
    gpu_kmp<<<std::ceil((float)comments_column->size / (float)TB), TB>>>(d_data, d_offsets, d_sizes,comments_column->size, d_matched_count, d_prefix,d_pattern, pattern.size());
    CUDACHKERR();

    // Retrieve results from GPU
    int gpu_matched_rows = 0;
    cudaMemcpy(&gpu_matched_rows, d_matched_count, sizeof(int), cudaMemcpyDeviceToHost);
    CUDACHKERR();

    std::cout << "Result from GPU: " << gpu_matched_rows << "\n";

    // Free device memory
    cudaFree(d_sizes);
    cudaFree(d_offsets);
    cudaFree(d_matched_count);
    cudaFree(d_data);
    cudaFree(d_pattern);
    cudaFree(d_prefix);

    // Free host memory
    delete[] prefix;

    return 0;
}
