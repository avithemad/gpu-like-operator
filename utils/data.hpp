#include <iostream>
#include <fstream>
#include <sstream>

#include <arrow/io/api.h>
#include <arrow/table.h>
#include <arrow/record_batch.h>
#include <arrow/array.h>
#include <parquet/arrow/reader.h>

namespace gpulike
{

    arrow::Status read_parquet(std::string path_to_file, std::shared_ptr<arrow::Table> &table)
    {
        arrow::MemoryPool *pool = arrow::default_memory_pool();
        // open file
        std::shared_ptr<arrow::io::RandomAccessFile> input;
        ARROW_ASSIGN_OR_RAISE(input, arrow::io::ReadableFile::Open(path_to_file));
        // initialize file reader
        std::unique_ptr<parquet::arrow::FileReader> arrow_reader;
        ARROW_RETURN_NOT_OK(parquet::arrow::OpenFile(input, pool, &arrow_reader));

        ARROW_RETURN_NOT_OK(arrow_reader->ReadTable(&table));

        return arrow::Status::OK();
    }

    std::shared_ptr<arrow::Table> getArrowTable(std::string path_to_file)
    {
        std::shared_ptr<arrow::Table> table;
        arrow::Status st = read_parquet(path_to_file, table);
        if (st != arrow::Status::OK())
        {
            std::cerr << st.ToString();
            return nullptr;
        }
        return table;
    }

    struct StringColumn
    {
        int *sizes;
        char **stringAddresses;
        char *data;
        StringColumn() {}
    };

    StringColumn *read_string_column(std::shared_ptr<arrow::Table> &table, const std::string &column)
    {

        // TODO: add all kinds of error handling
        auto arrow_col = table->GetColumnByName(column);
        StringColumn *result = new StringColumn();

        // calculate the size of data
        result->sizes = (int *)malloc(sizeof(int) * table->num_rows());
        result->stringAddresses = (char **)malloc(sizeof(char *) * table->num_rows());
        int data_size = 0;
        int j = 0;
        for (auto chunk : arrow_col->chunks())
        {
            auto string_arr = std::static_pointer_cast<arrow::LargeStringArray>(chunk);
            for (int i = 0; i < string_arr->length(); i++)
            {
                auto str = string_arr->GetString(i);
                data_size += str.size();
                result->sizes[j++] = str.size();
            }
        }
        result->data = (char *)malloc(sizeof(char) * data_size);
        j = 0;
        int k;
        char *straddr = result->data;
        for (auto chunk : arrow_col->chunks())
        {
            auto string_arr = std::static_pointer_cast<arrow::LargeStringArray>(chunk);
            for (int i = 0; i < string_arr->length(); i++)
            {
                auto str = string_arr->GetString(i);
                // std::cout << j << " " << str << "\n";
                result->stringAddresses[i] = straddr;
                straddr += str.size();
                for (auto c : str)
                {
                    result->data[j++] = c;
                }
            }
        }
        return result;
    }
    StringColumn* read_txt(std::string filepath) {
            std::ifstream file(filepath);
        if (!file.is_open()) {
            std::cerr << "Failed to open file: " << filepath << std::endl;
            return nullptr;
        }

        std::stringstream buffer;
        buffer << file.rdbuf(); // Read the file into the stringstream

        file.close(); // Close the file
        StringColumn *result = new StringColumn();
        int total_comments = 0;
        for (auto c: buffer.str()) {
            if (c == '\n') total_comments++;
        }
        result->sizes = (int*)malloc(sizeof(int)*total_comments);
        result->data = (char*)malloc(sizeof(char)*buffer.str().size());
        result->stringAddresses = (char**)malloc(sizeof(char*)*total_comments);
        int cur_size = 0, i=0, j=0;
        char* init = result->data;
        for (auto c: buffer.str()) {
            if (c == '\n') {
                result->sizes[i] = cur_size;
                result->stringAddresses[i] = init;
                cur_size=0;
                i++;
            } else {
                cur_size++;
            }
            init++;
            result->data[j] = c;
            j++;
        }
        std::cout << "Total comments: " << total_comments << std::endl;
        return result; // Return the contents as a string
    }
    int read_txt_size(std::string filepath) {
        std::ifstream file(filepath);
        if (!file.is_open()) {
            std::cerr << "Faile to open: " << filepath <<std::endl;
            return -1;
        }
        std::stringstream buffer;
        buffer << file.rdbuf(); // Read the file into the stringstream

        file.close(); // Close the file
        int result = 0;
        for (auto c: buffer.str()) {
            if (c == '\n') result++;
        }
        return result;
    }
}