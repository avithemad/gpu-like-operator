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
// Function to create bitmasks based on a pattern string with range support and negation handling
inline std::vector<std::vector<uint64_t>> createBitmasks(char* input, int& p_size) {
    std::vector<uint64_t> mask(BLOCKS_PER_MASK, 0);  // Initialize a mask with BLOCKS_PER_MASK 64-bit blocks
    std::vector<std::vector<uint64_t>> bitmasks;
    std::string s;  // Updated pattern string without ranges
    bool negation = false;
    int block_index, position_within_block, bit;

    for (int i = 0; i < p_size; i++) {
        s += input[i];  // Add character to the updated pattern
        if (input[i] == '[') {
            i++;
            if (input[i] == '^') {  // Check for negation
                negation = true;
                i++;
            }

            // Process the range or individual characters inside []
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

            if (negation) {
                // Complement the mask to invert the bitmask
                 for (int i = 0; i < BLOCKS_PER_MASK; i++) {
                    mask[i] = ~mask[i];
                }

                negation = false;  // Reset negation for the next mask
            }

            bitmasks.push_back(mask);  // Store the current mask
            mask.assign(BLOCKS_PER_MASK, 0);  // Reset for the next iteration
        }
    }

    // Update the pattern string and its size
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


int* compute_prefix(const std::string& pattern) {
    int m = pattern.size();
    int* prefix = new int[m];  
    int k = 0;
    prefix[0] = 0;
    int base=0;
    for (int i = 1; i < m; i++) {
        if(pattern[i]=='%'){
              base=i+1;
              i++;
              k=i;
              prefix[i]=i;
              continue;
          }
        while (k > base && pattern[k] != pattern[i]) {
            k = prefix[k - 1];
        }
        if (pattern[k] == pattern[i]) {
            k++;
        }
        prefix[i] = k;
    }

    return prefix; 
}