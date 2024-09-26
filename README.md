# GPU Like operator for database queries

This repo is to experiment how best we can accelerate the like operator in SQL queries using GPUs.

## Dependencies

- CMAKE 3.27.0
- nvidia-cuda-toolkit, nvcc must support c++17 standard.

## Building


Create a build directory

``
mkdir build
cd build
``

Configure CMake

``
cmake  ..
``
Invoke cmake build

``
cmake --build .
``