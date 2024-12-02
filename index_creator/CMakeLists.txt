
add_executable(index-creator index_creator.cpp)

target_include_directories(index-creator PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../utils)

