#include "data.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <map>
#include <algorithm>

std::string key_to_charseq(int k) {
  std::string res;
  res.push_back(k/256);
  res.push_back(k%256);
  return res;
}
bool valid(char c)
{
  return (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || (c >= 'A' && c <= 'Z');
}

int main(int argc, char *argv[]){
  std::string file_path = argv[1];
  gpulike::StringColumn *col  = gpulike::read_txt(file_path);

  std::vector<int> seq_count(256*256, 0);
  for (int i=0; i<col->size; i++) {
    int offset = col->offsets[i];
    for (int j=1; j<col->sizes[i]; j++) {
      char prev = col->data[offset + j - 1];
      char curr = col->data[offset + j];
      if (valid(prev) && valid(curr)) {
        int k = prev*256 + curr;
        seq_count[k] = 1;
      }
    }
  }
  for (int i=1; i<seq_count.size(); i++) seq_count[i] += seq_count[i-1];
  std::cout << "Total bitmaps: " << seq_count[seq_count.size() - 1] << std::endl;
  std::vector<std::vector<uint64_t>> bitmaps(seq_count[seq_count.size()-1] + 1,
                                             std::vector<uint64_t>(std::ceil((float)col->size / 64.), 0));


  for (int i=0; i<col->size; i++) {
    int offset = col->offsets[i];
    for (int j=1; j<col->sizes[i]; j++) {
      char prev = col->data[offset + j - 1];
      char curr = col->data[offset + j];
      if (valid(prev) && valid(curr)) {
        int k = prev*256 + curr;
        bitmaps[seq_count[k]][i / 64] |= ((uint64_t)1 << (63 - i%64));
      }
    }
  }
  std::map<int, int> bmind_to_charkey;
  for (int i=seq_count.size() - 1; i>=0; i--) {
    bmind_to_charkey[seq_count[i]] = i;
  } 
  std::vector<std::pair<float, std::string>> char_seq_selectivity;
  for (int i=1; i<bitmaps.size(); i++) {
    int res = 0;
    for (auto e: bitmaps[i]) {
      while (e!=0) {
        if (e & 0x1) res++;
        e >>= 1;
      }
    } 
    char_seq_selectivity.push_back(std::make_pair((float)res/(float)col->size, key_to_charseq(bmind_to_charkey[i])));
  }
  std::sort(char_seq_selectivity.begin(), char_seq_selectivity.end());
  float avg = 0.;
  for (auto e: char_seq_selectivity) {
    std::cout << e.second << " : " << e.first << std::endl;
    avg += e.first;
  }
  avg /= char_seq_selectivity.size();
  std::cout << "Avg speed up expected: " << 1.0/avg << std::endl;
}