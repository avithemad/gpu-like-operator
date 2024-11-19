#include "data.hpp"
#include <iostream>
#include <cassert>
#include <vector>
#include "cudautils.cuh"
#include <cstring> 
#define BITS_PER_WORD 64
#define BITS_IN_MASK 256
#define NUM_BLOCKS (BITS_IN_MASK / BITS_PER_WORD)  // 256 bits / 64 bits = 4 blocks



int cpu_kmp(gpulike::StringColumn* comments_column, std::string pattern, int* prefix){
    // q denotes the lenght of matched string
    int q=0,matched_rows = 0;
    // base saves the progress of matched subpatterns split by %
    int base=0;
    for(int i=0;i<comments_column->size;i++){
        q=0,base=0;
    for (int j=0; j<(comments_column->sizes[i]); j++) {
        if(pattern[q]=='%')
            {
                q++;
                base=q;
            }
        while(q>base && pattern[q]!=comments_column->data[comments_column->offsets[i]+j])
            {
                q=prefix[q-1];
            }
        if(pattern[q]==comments_column->data[comments_column->offsets[i]+j])
            q++;
        if(q==pattern.size()){
            matched_rows++;
            // std::cout<<"found at "<<j-q;
            break;
        }
    }
    }
    return matched_rows;
}
__global__ void gpu_kmp(char* data, int* offsets, int* sizes, size_t table_size, int* matched_count,int* prefix, char* pattern,int p_size) {
    int tid = threadIdx.x + blockDim.x*blockIdx.x;
    if(tid >= table_size) return;
   
    int q=0;
    // const char* pattern = "a";
    int m=0,per=0,p=0,b=0,base=0;
     for (int j=0; j<sizes[tid]; j++) {
      if(pattern[q]=='%')
            {
                q++;
                base=q;
            }
        while(q>base && pattern[q]!=data[offsets[tid]+j])
            {
                q=prefix[q-1];
            }
        if(pattern[q]==data[offsets[tid]+j])
            q++;
        if(q==p_size){
            atomicAdd(matched_count, 1);
            // std::cout<<"found at ";
            break;
        }
     }

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
__global__ void gpu_brute_force_Purr(char* data, int* offsets, int* sizes, size_t table_size, int* matched_count,u_int64_t* bitmasks1d, char* pattern) {
  // const char* pattern = "%are%the%";
  // size_t p_size = 9;
  // int m=0,per=0,p=3,b=0;
  size_t p_size = 1;
  int m=0,per=0,p=0,b=0;
  // printf("hello");
  // m for store the index upto which subpattern matched 
  // p for not including the count of %
  // per to account for the increment in k dues to %
  int tid = threadIdx.x + blockDim.x*blockIdx.x;
  if (tid >= table_size) return;
  // printf("GPU:%s\n", pattern);

  // limit checking of string to total length -pattern length
  for (int j=0; j<sizes[tid] - p_size + p + 1; j++) {
    bool matched = true;b=0;
    for (int k=0+m; k<p_size; k++) {
      if(pattern[k]=='%'){
            m=k+1;
            per++;
            continue;
        }

      // implementation of [] using 1dbitmasks
      if(pattern[k]=='['){
          int num=data[offsets[tid]+j+k-per]/BITS_PER_WORD;
          int den=data[offsets[tid]+j+k-per]%BITS_PER_WORD;
            if(!(bitmasks1d[b*4 +num]>>(den) & 1)){
            printf("%d\n",(int)(bitmasks1d[b*4 +num]>>(den) & 1));
                matched = false;
                break;
            }
            b++;
            continue;
        }


      if (data[offsets[tid] + k + j-per] != pattern[k]) {
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

//bitmasks vector creation
std::vector<std::vector<uint64_t>> createBitmasks(std::string& input){
  std::vector<uint64_t> mask(NUM_BLOCKS, 0);  // Initialize a mask with 4 64-bit blocks
  std::vector<std::vector<uint64_t>> bitmasks; // Vector to store multiple 256-bit masks
    std::string s;
    int block_index,bit_position,bit;
    for (int i = 0; i < input.size(); i++) {
        s += input[i];
        if (input[i] == '[') {
            i++;
            while (input[i] != ']') {
                if (input[i] == '-') {
                    // Handle the range case (e.g., a-c)
                    for (int j = input[i - 1]; j <= input[i + 1]; j++) {
                        block_index = j / BITS_PER_WORD; // Determine which 64-bit block the bit belongs to
                        bit_position = j % BITS_PER_WORD; // Determine the position within that block
                        mask[block_index] |= (1ULL << bit_position); // Set the bit in the correct block
                    }
                    i++;
                } else {
                    // Set the individual bit for non-range characters
                    bit = input[i];
                    block_index = bit / BITS_PER_WORD;
                    bit_position = bit % BITS_PER_WORD;
                    mask[block_index] |= (1ULL << bit_position);
                }
                i++;
            }

            bitmasks.push_back(mask); // Store the current mask
            mask = std::vector<uint64_t>(NUM_BLOCKS, 0); // Reset the mask for the next iteration
        }
    }
    // to display bitmasks of []s
    // for(auto a:bitmasks){
    //   for(int i=0;i<64;i++)
    //     std::cout<<(int)(a[1]>>i&1);
    //   std::cout<<std::endl;
    // }
    input = s;
    return bitmasks;
}
std::vector<std::string> splitByPercentage(const std::string& input) {
    std::vector<std::string> patterns;
    std::string currentPattern;

    for (char c : input) {
        if (c == '%') {
            if(currentPattern!="")
            // Add the current pattern to the vector, even if it's empty
            patterns.push_back(currentPattern);
            currentPattern.clear(); // Clear for the next Pattern
        } else {
            currentPattern += c; // Build the current Pattern
        }
    }
    
    // Add the last Pattern after the loop (if it's not empty)
    if(currentPattern!="")
    patterns.push_back(currentPattern);
    
    return patterns;
}
// didnt use split subpatterns in vector
int cpu_brute_force_noVec(gpulike::StringColumn* comments_column, std::string pattern, std::vector<std::vector<uint64_t>> bitmasks) {
  std::vector<std::string> patterns;int matched_rows = 0;

  
  int m=0,per=0,p=0;
  //current mask position
  int b=0;
  //need % count
  for(char c:pattern){
    if(c=='%') p++;
  }
  // cpu side matching
  for (int i=0; i<comments_column->size; i++) {
    m=0;per=0;
    // per to account for the % to be not included in pattern size of matching string patterns
    for (int j=0; j<(comments_column->sizes[i] - pattern.size()+p+1 ); j++) {
      bool matched = true;
      b=0;
      for (int k=0+m; k<(pattern.size()); k++) {
        // need to account for when % is matched we set our start point from the next string pattern
        if(pattern[k]=='%'){
            m=k+1;
            per++;
            continue;
        
        }
        
        // implementation of [] range and set using bitmasks
        if(pattern[k]=='['){
          int num=comments_column->data[comments_column->offsets[i]+j+k-per]/BITS_PER_WORD;
          int den=comments_column->data[comments_column->offsets[i]+j+k-per]%BITS_PER_WORD;
            if(!(bitmasks[b][num]>>den & 1)){
                matched = false;
                break;
            }
            b++;
            continue;
        }

        if (comments_column->data[comments_column->offsets[i]+j+k-per]!=pattern[k]) 
        {
          matched = false;
          break;  
        }

      }
      if (matched) {
        // check for next pattern
        matched_rows++;
        break;}
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

    // Precompute KMP prefix array
    int* prefix = compute_prefix(pattern);

    // CPU KMP
    int cpu_matched_rows = cpu_kmp(comments_column, pattern, prefix);
    std::cout << "Total matched rows in CPU KMP: " << cpu_matched_rows << "\n";

    // CPU Brute Force
    std::vector<std::vector<uint64_t>> bitmasks = createBitmasks(pattern);
    cpu_matched_rows = cpu_brute_force_noVec(comments_column, pattern, bitmasks);
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
    gpu_kmp<<<std::ceil((float)comments_column->size / (float)TB), TB>>>(d_data, d_offsets, d_sizes,
                                                                         comments_column->size, d_matched_count, d_prefix,
                                                                         d_pattern, pattern.size());
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
