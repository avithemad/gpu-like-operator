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
        StringColumn() {}
    };

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
        if (total_comments == 0) {
            std::cout << "No data in given file: " << filepath << std::endl;
            return nullptr;
        }
        result->sizes = (int*)malloc(sizeof(int)*total_comments);
        result->data = (char*)malloc(sizeof(char)*buffer.str().size());
        result->offsets = (int*)malloc(sizeof(int)*total_comments);

        int cur_size = 0, i=0, j=0;
        result->offsets[0] = 0;

        for (auto c: buffer.str()) {
            if (c == '\n') {
                result->sizes[i] = cur_size;
                cur_size=0;
                i++;
                if (i < total_comments) {
                    result->offsets[i] = result->sizes[i-1] + result->offsets[i-1];
                }
            } else {
                cur_size++;
                result->data[j] = c;
                j++;
            }
        }
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