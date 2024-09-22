#include <iostream>

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
}