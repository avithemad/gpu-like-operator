#include "data.hpp"
#include <iostream>

int main() {
  std::string dbDir = "/media/ajayakar/space/src/tpch/data/tables/scale-1.0/";
  std::string lineitem_file = dbDir + "lineitem.parquet";

  auto lineitem_table = gpulike::getArrowTable(lineitem_file);

  std::cout << lineitem_table->schema()->ToString();

  gpulike::StringColumn* comments_column = gpulike::read_string_column(lineitem_table, "comments");

  std::cout << comments_column->data;
}