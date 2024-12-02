
add_executable(fft fft.cu)
add_executable(batched-fft batched-fft.cu)
add_compile_options(batched-fft -g -O3)


target_include_directories(fft PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../utils)
target_include_directories(batched-fft PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../utils)


target_link_libraries(fft PRIVATE "/usr/local/cuda-12.6/lib64/libcufft.so")
target_link_libraries(batched-fft PRIVATE "/usr/local/cuda-12.6/lib64/libcufft.so")
