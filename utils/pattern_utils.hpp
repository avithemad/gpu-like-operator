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


