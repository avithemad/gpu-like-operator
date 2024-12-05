#pragma once

#include <vector>
#include <string>
#include <cstdint>
#include <cstring>
#include <iostream>

// Constants for bit manipulation
#define BITS_PER_BLOCK 64
#define BITS_IN_MASK 256
#define BLOCKS_PER_MASK (BITS_IN_MASK / BITS_PER_BLOCK)

// Function to create bitmasks based on a pattern string with range support
inline std::vector<std::vector<uint64_t>> createBitmasks(char* input, int& p_size) {
    std::vector<uint64_t> mask(BLOCKS_PER_MASK, 0);  // Initialize a mask with 4 64-bit blocks
    std::vector<std::vector<uint64_t>> bitmasks;
    std::string s;
    int block_index, position_within_block, bit;

    for (int i = 0; i < p_size; i++) {
        s += input[i];
        if (input[i] == '[') {
            i++;
            while (input[i] != ']') {
                if (input[i] == '-') {
                    // Handle range case (e.g., a-c)
                    for (int str_index = input[i - 1]; str_index <= input[i + 1]; str_index++) {
                        block_index = str_index / BITS_PER_BLOCK;
                        position_within_block = str_index % BITS_PER_BLOCK;
                        mask[block_index] |= (1ULL << position_within_block);
                    }
                    i++;
                } else {
                    // Set individual bit for non-range characters
                    bit = input[i];
                    block_index = bit / BITS_PER_BLOCK;
                    position_within_block = bit % BITS_PER_BLOCK;
                    mask[block_index] |= (1ULL << position_within_block);
                }
                i++;
            }
            bitmasks.push_back(mask);  // Store the current mask
            mask.assign(BLOCKS_PER_MASK, 0);  // Reset for next iteration
        }
    }

    std::strcpy(input, s.c_str());
    p_size = s.size();
    return bitmasks;
}

// Function to split a pattern by the '%' character
inline std::vector<std::string> splitByPercentage(const std::string& input) {
    std::vector<std::string> patterns;
    std::string currentPattern;

    for (char c : input) {
        if (c == '%') {
            if (!currentPattern.empty()) {
                patterns.push_back(currentPattern);
            }
            currentPattern.clear();
        } else {
            currentPattern += c;
        }
    }
    if (!currentPattern.empty()) {
        patterns.push_back(currentPattern);
    }
    return patterns;
}

// Function to count occurrences of '%' in the pattern
inline int count_per(const std::string& pattern) {
    int count = 0;
    for (char c : pattern) {
        if (c == '%') {
            count++;
        }
    }
    return count;
}

struct preprocess_data {
  std::string pattern;
  std::vector<int> sp_sizes;
  std::vector<int64_t> wildcards;
  std::vector<std::vector<int>> prefix_tables;
  std::vector<int> prefix_tables_gpu;
  std::vector<int> prefix_tables_gpu_sizes;
  preprocess_data(std::string pattern, std::vector<int> sp_sizes, std::vector<int64_t> wildcards, std::vector<std::vector<int>> prefix_tables) :
    pattern(pattern), sp_sizes(sp_sizes), wildcards(wildcards), prefix_tables(prefix_tables) {
      for (auto pi : prefix_tables) {
        prefix_tables_gpu_sizes.push_back(pi.size());
        for (auto e: pi) prefix_tables_gpu.push_back(e);
      }
    }
  void print() {
    std::cout << "preprocessed pattern: " << pattern << "\n";
    std::cout << "subpattern count: " << sp_sizes.size() << "\n";
    for (auto e: sp_sizes) {
      std::cout << e << " ";
    } 
    std::cout << "\n";
    std::cout << "bitmasks\n";
    for (auto bm: wildcards) {
      printf("%lx\t", bm);
    }
    std::cout << "\nkmp prefixes\n";
    for (auto pi: prefix_tables) {
        for (auto e: pi) {
            std::cout << e << "\t";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
    std::cout << "prefix_tables_gpu\n";
    for (auto e: prefix_tables_gpu) std::cout << e << "\t";
    std::cout << "\n";
    std::cout << "prefix_tables_gpu_sizes\n";
    for (auto e: prefix_tables_gpu_sizes) std::cout << e << "\t";
    std::cout << "\n";
  }
};


void compute_prefix_table(std::string pattern, int *prefix_table, int n) {
    for (int i = 1; i < n; ++i) {
        int j = prefix_table[i - 1]; // Initialize j with the previous prefix function value
        // Iterate backwards through possible lengths for the prefix/suffix
        while (j > 0 && pattern[j] != pattern[i]) {
            j = prefix_table[j - 1]; // Update j based on the previous prefix function value
        }
        // Check if the substrings are equal
        if (pattern[j] == pattern[i]) {
            ++j; // If equal, update the value of j
        }
        prefix_table[i] = j; // Update the prefix function value at the current position
    }
}


preprocess_data preprocess_pattern(char *pattern, int p_size) {
  std::string result = "";
  std::vector<int> sp_sizes;
  std::vector<int64_t> wildcards(0);
  int prev_per = -1;
  int sz = 0;
  for (int i=0; i < p_size; i++) {
    if (pattern[i] == '[') {
      result.push_back('[');
      std::vector<int64_t> bitmask(4, 0x0);
      i++;
      bool nt = false;
      int initial_lbrace = i - 1;
      if (pattern[i] == '^') {nt = true; i++; initial_lbrace--;}
      int j = i, final_rbrace = p_size; // search for the final ]
      bool rbrace_found = false;
      while (( !rbrace_found || pattern[j]!='[') && j < p_size) {
        if (pattern[j] == ']') {final_rbrace = j; rbrace_found = true;}
        j++;
      }
      while(i < final_rbrace) {
        if (pattern[i] == '-' && i - 1 != initial_lbrace && i+1 != final_rbrace && pattern[i-1] < pattern[i+1]) { // check for ranges
          for (char c = pattern[i-1]; c <= pattern[i+1]; c++) {
            int idx = c/64, offset = c%64;
            bitmask[idx] |= ((int64_t)1 << offset);    
          }
          i++;
          continue;
        }
        std::cout << pattern[i] << "\n";
        int idx = pattern[i]/64, offset = pattern[i]%64;
        bitmask[idx] |= ((int64_t)1 << offset);
        i++;
      }
      sz++;
      if (!nt)
        for (auto e: bitmask) wildcards.push_back(e);
      else 
        for (auto e: bitmask) wildcards.push_back(0xFFFFFFFFFFFFFFFF ^ e);
    } else if (pattern[i] == '%') {
      sp_sizes.push_back(sz);
      prev_per = i;
      sz =0;
    } else {
      result.push_back(pattern[i]);
      sz++;
    }
  }
  if (sp_sizes.size() == 0) {
    sp_sizes.push_back(result.size());
    sp_sizes.push_back(-1);
  } else {
    sp_sizes.push_back(p_size - prev_per - 1);
  }
  std::vector<std::vector<int>> prefix_tables;
  if (sp_sizes.size() > 2) {
    std::vector<std::string> subpatterns;
    int i = sp_sizes[0];
    for (int k=1; k<sp_sizes.size()-1; k++) {
        std::string sp;
        for (int j=i; j<i+sp_sizes[k]; j++) {
            if (result[j] == '_' || result[j] == '[') break; // consider only first part of the pattern
            sp.push_back(result[j]);
        }
        subpatterns.push_back(sp);
        i += sp_sizes[k];
    }
    for (int i=1; i<sp_sizes.size() - 1; i++) {
        std::vector<int> pi(subpatterns[i-1].size(), 0);
        compute_prefix_table(subpatterns[i-1], pi.data(), subpatterns[i-1].size());
        prefix_tables.push_back(pi);
    }
  }
  return preprocess_data(result, sp_sizes, wildcards, prefix_tables);
}

