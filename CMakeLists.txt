cmake_minimum_required(VERSION 3.16.3)

project(GPULIKE VERSION 0.0.1 LANGUAGES CXX CUDA)

add_subdirectory(impl)
add_subdirectory(fft-impl)
add_subdirectory(index_creator)
