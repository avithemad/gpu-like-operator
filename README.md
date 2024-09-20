# GPU Like operator for database queries

This repo is to experiment how best we can accelerate the like operator in SQL queries using GPUs.

## Dependencies

The project is implemented for CUDA capable GPUs. Make sure you have the driver installed with nvidia cuda toolkit as well. 
Refer https://docs.nvidia.com/cuda/cuda-installation-guide-linux/.

For reading test data, we use the Arrow library, which can be build from source https://github.com/apache/arrow/blob/main/docs/source/developers/cpp/building.rst.
Use the `-DARROW_PARQUET=ON` while configuring build for arrow to have support for parquet files. 

## Building

Make sure dependencies are installed.

Create a build directory

``
mkdir build
cd build
``

Configure CMake

``
cmake -DARROW_DIR=<path to arrow installation> ..
``
Invoke cmake build


``
cmake --build .
``