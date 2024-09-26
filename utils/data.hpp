#pragma once

#include <iostream>
#include <fstream>
#include <sstream>
namespace gpulike
{
  struct StringColumn
  {
    int *sizes;
    int *offsets;
    char *data;
    int size;
    StringColumn() {}
  };

  struct StringColumnPivoted
  {
    char **data;
    int warp_size;
    int *max_lens;
    int size;
    StringColumnPivoted() {}
  };

  StringColumnPivoted *convert_to_transpose(StringColumn *column_data, int warp_size)
  {
    int total_size = std::ceil((float)column_data->size / (float)warp_size);
    auto result = new StringColumnPivoted();

    result->data = (char **)malloc(sizeof(char *) * total_size);
    result->size = total_size;
    result->warp_size = warp_size;
    result->max_lens = (int *)malloc(sizeof(int) * total_size);

    // consider data chunk by chunk
    for (int i = 0; i < total_size; i++)
    {
      // get the max size in this warp
      int max_len = 0;
      for (int j = 0; j < warp_size; j++)
      {
        max_len = std::max(max_len, column_data->sizes[i * warp_size + j]);
      }
      result->data[i] = (char *)malloc(sizeof(char) * warp_size * max_len);
      memset(result->data[i], '\0', sizeof(char) * warp_size * max_len);
      result->max_lens[i] = max_len;
      for (int j = 0; j < warp_size; j++)
      {
        int offset = column_data->offsets[i * warp_size + j];
        for (int k = 0; k < column_data->sizes[i * warp_size + j]; k++)
        {
          result->data[i][k * warp_size + j] = column_data->data[offset + k];
        }
      }
    }
    return result;
  }
  void print_pivoted(StringColumnPivoted *comments_pivoted)
  {
    for (int i = 0; i < comments_pivoted->size; i++)
    {
      for (int j = 0; j < comments_pivoted->max_lens[i] * comments_pivoted->warp_size; j++)
      {
        if (j % comments_pivoted->warp_size == 0)
          std::cout << "\n";
        if (comments_pivoted->data[i][j] == '\n' || comments_pivoted->data[i][j] == '\0')
          std::cout << " ";
        else
          std::cout << comments_pivoted->data[i][j];
      }
      std::cout << "\n\n";
    }
  }

  StringColumn *read_txt(std::string filepath)
  {
    std::ifstream file(filepath);
    if (!file.is_open())
    {
      std::cerr << "Failed to open file: " << filepath << std::endl;
      return nullptr;
    }

    std::stringstream buffer;
    buffer << file.rdbuf(); // Read the file into the stringstream

    file.close(); // Close the file
    StringColumn *result = new StringColumn();
    int total_comments = 0;
    for (auto c : buffer.str())
    {
      if (c == '\n')
        total_comments++;
    }
    if (total_comments == 0)
    {
      std::cout << "No data in given file: " << filepath << std::endl;
      return nullptr;
    }
    result->sizes = (int *)malloc(sizeof(int) * total_comments);
    result->data = (char *)malloc(sizeof(char) * buffer.str().size());
    result->offsets = (int *)malloc(sizeof(int) * total_comments);

    int cur_size = 0, i = 0, j = 0;
    result->offsets[0] = 0;

    for (auto c : buffer.str())
    {
      if (c == '\n')
      {
        result->sizes[i] = cur_size;
        cur_size = 0;
        i++;
        if (i < total_comments)
        {
          result->offsets[i] = result->sizes[i - 1] + result->offsets[i - 1];
        }
      }
      else
      {
        cur_size++;
        result->data[j] = c;
        j++;
      }
    }
    result->size = i;
    return result; // Return the contents as a string
  }
}