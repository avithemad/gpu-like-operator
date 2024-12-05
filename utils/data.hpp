#pragma once

#include <iostream>
#include <fstream>
#include <sstream>
#include <string.h>
#include <math.h>
namespace gpulike
{
  /**
   * First chunk will be a 2D character matrix with warp_size * max_lens, number of elements.
   * ith string can be accesses as follows
   * for j = 0 to max_lens[warp_size]
   *   data[i/warp_size][j*warp_size + i%warp_size]
   */
  struct StringColumnPivoted
  {
    char **data;
    int warp_size;
    int *max_lens;
    int size;
    StringColumnPivoted() {}
  };


  struct StringColumn
  {
    char *data;
    int *sizes;
    int64_t *offsets;
    int64_t size;
    StringColumn() {}
  };
  struct StringColumnPivotedK {
    char *data;
    int max_len;
    int64_t size;
    StringColumnPivotedK() {}
  };

  StringColumnPivotedK *convert_to_pivotedk(StringColumn *col) {
    int max_len = 0;
    for (int i=0; i<col->size; i++) max_len = std::max(max_len, col->sizes[i]);
    std::cout << "pivot conv: total allocation = " << sizeof(char)*max_len*col->size / (1024 * 1024) << "MB" << "\n";
    StringColumnPivotedK* res = new StringColumnPivotedK();
    res->data = (char*)malloc(sizeof(char)*max_len*col->size);
    res->size = col->size;
    res->max_len = max_len;
    memset(res->data, 0, sizeof(char)*max_len*col->size);
    for (int64_t i=0; i<col->size; i++) {
      for (int64_t j=0; j<col->sizes[i]; j++) {
        res->data[j*col->size + i] = col->data[col->offsets[i] + j];
      }
    }
    return res;
  }

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
  void print_pivoted_to_normal(StringColumnPivoted *comments_pivoted)
  {
    for (int i = 0; i < comments_pivoted->size; i++)
    {
      for (int j = 0; j < comments_pivoted->warp_size; j++)
      {
        for (int k = 0; k < comments_pivoted->max_lens[i]; k++)
        {
          char c = comments_pivoted->data[i][k * comments_pivoted->warp_size + j];
          if (c == '\0')
          {
            break;
          }
          std::cout << c;
        }
        std::cout << "\n";
      }
    }
  }

  StringColumn *read_txt(std::string filepath)
  {
    std::cout << "Reading file " << filepath << "\n";

    FILE *fptr = NULL;
    fptr = fopen(filepath.c_str(), "r");
    StringColumn *result = new StringColumn();

    long long total_comments = 0, total_chars = 0;
    char c;
    while ((c = fgetc(fptr))!=EOF)
    {
      if (c == '\n')
        total_comments++;
      else 
        total_chars++;
    }
    fclose(fptr);
    std::cout << "total lines: " << total_comments << "\n";
    std::cout << "total characters: " << total_chars << "\n";
    std::cout << "average chars/line: " << total_chars/total_comments << "\n";
    if (total_comments == 0)
    {
      std::cout << "No data in given file: " << filepath << std::endl;
      return nullptr;
    }
    result->sizes = (int *)malloc(sizeof(int) * total_comments);
    result->data = (char *)malloc(sizeof(char) * total_chars);
    result->offsets = (int64_t *)malloc(sizeof(int64_t) * total_comments);

    int64_t cur_size = 0, i = 0, j = 0;
    result->offsets[0] = 0;
    fptr = fopen(filepath.c_str(), "r");
    while ((c = fgetc(fptr))!=EOF) {
      if (c == '\n') {
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
    fclose(fptr);

    return result; // Return the contents as a string
  }
}