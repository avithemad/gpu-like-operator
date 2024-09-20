
add_executable(brute-force brute-force.cu)
target_include_directories(brute-force PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../utils "/media/ajayakar/space/tools/arrow/include")
target_link_directories(brute-force PRIVATE "/media/ajayakar/space/tools/arrow/lib")
target_link_libraries(brute-force PRIVATE arrow parquet)
