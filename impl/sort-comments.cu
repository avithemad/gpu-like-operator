#include "data.hpp"
#include <map>

#include <iostream>
int main(int argc, char* argv[]) {
  std::string txt_file = argv[1];
  gpulike::StringColumn* col = gpulike::read_txt(txt_file);

  std::multimap<int, std::string> mp;
  for (int i=0; i<col->size; i++) {
    std::string s;
    for (int j=0; j<col->sizes[i]; j++) {
      s.push_back(col->data[col->offsets[i] + j]);
    }
    mp.insert(std::pair<int, std::string>(s.size(), s));
  }
  for (auto &p: mp) {
    std::cout << p.second << std::endl;
  }
}