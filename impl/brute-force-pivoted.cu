#include "data.hpp"
#include <iostream>
#include <cassert>
#include "cudautils.cuh"

int main(int argc, char *argv[])
{
  if (argc < 2)
  {
    std::cout << "Please provide path to string column file. eg: ./brute-force /media/db/comments.txt";
  }
  std::string txt_file = argv[1];

  gpulike::StringColumn *comments_column = gpulike::read_txt(txt_file);
  if (comments_column == nullptr)
  {
    std::cout << "Unable to read comments columns, possibly no data in the file\n";
    exit(0);
  }

  int TB = 32;
  gpulike::StringColumnPivoted *comments_pivoted = gpulike::convert_to_transpose(comments_column, TB);
  // gpulike::print_pivoted(comments_pivoted);

  // allocate gpu resident data
  // char** d_pivoted_data;
  // int* d_pivoted_lengths;
  // cudaMalloc(&d_pivoted_data, sizeof(char *) * comments_pivoted->size);
  // for (int i=0; i<comments_pivoted->size; i++) {
  //   char* d_block_data;
  // }
}
