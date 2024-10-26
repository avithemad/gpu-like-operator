#include "data.hpp"
#include <iostream>
#include <cassert>
#include "cudautils.cuh"

__global__ void gpu_kmp_basic(char *data,
                              int *offsets,
                              int *sizes,
                              size_t table_size,
                              char *pattern,
                              int p_size,
                              int *prefix_table,
                              int *matched_count)
{
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid >= table_size)
    return;

  // limit checking of string to total length -pattern length
  int i = 0, j = 0;
  // printf("%d, size: %d\n", tid, sizes[tid]);
  while (i < (sizes[tid]))
  {
    if (pattern[j] == data[offsets[tid] + i])
    {
      j++;
      i++;
      if (j == p_size)
      {
        atomicAdd(matched_count, 1);
        return;
      }
    }
    else
    {
      if (j != 0)
        j = prefix_table[j];
      else
        i++;
    }
  }
}

__global__ void gpu_kmp_pivoted(char **data,
                                int *max_lens,
                                char *pattern,
                                int p_size,
                                int *prefix_table,
                                int *res)
{
  int i = 0, j = 0;
  while (i < max_lens[blockIdx.x])
  {
    if (pattern[j] == data[blockIdx.x][i * blockDim.x + threadIdx.x])
    {
      j++;
      i++;
      if (j == p_size)
      {
        atomicAdd(res, 1);
        return;
      }
    }
    else
    {
      if (j != 0)
        j = prefix_table[j];
      else
        i++;
    }
  }
}

int cpu_brute_force(gpulike::StringColumn *comments_column, std::string pattern)
{
  int matched_rows = 0;
  // cpu side matching
  for (int i = 0; i < comments_column->size; i++)
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
        matched_rows++;
        break;
      }
    }
  }
  return matched_rows;
}
// compute prefix table for KMP in the host
void compute_prefix_table(const char *pattern, int p_size, int *prefix_table)
{
  for (int i = 0; i < p_size; i++)
  {
    if (pattern[0] != pattern[i] || i == 0)
      prefix_table[i] = 0;
    else
    {
      int j = 0;
      while (pattern[j] == pattern[i])
      {
        prefix_table[i] = j;
        j++;
        i++;
      }
      i--;
    }
  }
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

  const char *pattern = argv[2];
  int cpu_matched_rows = cpu_brute_force(comments_column, pattern);

  int *d_sizes, *d_matched_count, *d_res2;
  int *d_offsets;
  char *d_data;
  cudaMalloc(&d_sizes, sizeof(int) * comments_column->size);
  cudaMalloc(&d_matched_count, sizeof(int));
  cudaMalloc(&d_res2, sizeof(int));
  cudaMalloc(&d_offsets, sizeof(int) * comments_column->size);
  cudaMalloc(&d_data, sizeof(char) * data_size);

  cudaMemcpy(d_sizes, comments_column->sizes, sizeof(int) * comments_column->size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_offsets, comments_column->offsets, sizeof(int) * comments_column->size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_data, comments_column->data, sizeof(char) * data_size, cudaMemcpyHostToDevice);
  cudaMemset(d_matched_count, 0, sizeof(int));
  cudaMemset(d_res2, 0, sizeof(int));
  CUDACHKERR();

  int TB = 256;
  int p_size = 0;
  for (int i = 0; pattern[i] != '\0'; i++)
    p_size++;
  char *d_pattern;
  cudaMalloc(&d_pattern, sizeof(char) * p_size);
  cudaMemcpy(d_pattern, pattern, sizeof(char) * p_size, cudaMemcpyHostToDevice);
  int *prefix_table = (int *)malloc(sizeof(int) * p_size);
  compute_prefix_table(pattern, p_size, prefix_table);
  std::cout << "prefix table:\n";
  for (int i = 0; i < p_size; i++)
    std::cout << pattern[i] << "\t";
  std::cout << "\n";
  for (int i = 0; i < p_size; i++)
    std::cout << prefix_table[i] << "\t";
  std::cout << "\n";
  int *d_prefix_table;
  cudaMalloc(&d_prefix_table, sizeof(int) * p_size);
  cudaMemcpy(d_prefix_table, prefix_table, sizeof(int) * p_size, cudaMemcpyHostToDevice);
  gpu_kmp_basic<<<std::ceil((float)comments_column->size / (float)TB), TB>>>(d_data, d_offsets, d_sizes, comments_column->size, d_pattern, p_size, d_prefix_table, d_matched_count);
  CUDACHKERR();
  int gpu_matched_rows = 0;
  cudaMemcpy(&gpu_matched_rows, d_matched_count, sizeof(int), cudaMemcpyDeviceToHost);
  CUDACHKERR();



  assert(gpu_matched_rows == cpu_matched_rows);

  std::cout << "Result from GPU: " << gpu_matched_rows << "\n";

  std::cout << "Now running the pivoted kernel\n";
  gpulike::StringColumnPivoted *col_piv = gpulike::convert_to_transpose(comments_column, TB);
  char **d_data_piv, **h_data;
  int *d_max_lens;
  h_data = (char**)malloc(sizeof(char*)*col_piv->size);
  char *d_char_data;
  int total_data = 0;
  for (int i=0; i<col_piv->size; i++) total_data += col_piv->max_lens[i];
  cudaMalloc(&d_char_data, sizeof(char)*total_data*TB);
  for (int i=0; i<col_piv->size; i++) {
    h_data[i] = (i > 0 ? h_data[i-1] + col_piv->max_lens[i-1]*TB : d_char_data);
    cudaMemcpy(h_data[i], col_piv->data[i], sizeof(char)*col_piv->max_lens[i]*TB, cudaMemcpyHostToDevice); 
  }
  cudaMalloc(&d_data_piv, sizeof(char*)*col_piv->size);
  cudaMemcpy(d_data_piv, h_data, sizeof(char*)*col_piv->size, cudaMemcpyHostToDevice);
  cudaMalloc(&d_max_lens, sizeof(int)*col_piv->size);
  cudaMemcpy(d_max_lens, col_piv->max_lens, sizeof(int)*col_piv->size, cudaMemcpyHostToDevice);
  CUDACHKERR();

  gpu_kmp_pivoted<<<col_piv->size, TB>>>(d_data_piv, d_max_lens, d_pattern, p_size, d_prefix_table, d_res2 );
  CUDACHKERR();
  cudaMemcpy(&gpu_matched_rows, d_res2, sizeof(int), cudaMemcpyDeviceToHost);
  std::cout << "Result from pivoted layout: " << gpu_matched_rows << "\n";
}