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

## Running queries for like operator

The `final_impl`, has the source code for all the final implementations of like operator including
- Brute force, KMP, KMP step for normal layout: `main.cu`
- Brute force, KMP, KMP step for pivoted layout: `main_pivoted.cu`
- Bigram index in `main_pindex.cu`

Use the following command to run the LIKE operator with pattern

```
<src>/build/final_impl/main <path to column file> <pattern>

//example
<src>/build/final_impl/main /Data/tables/lineitem/comments.col %express%dependencies%
<src>/build/final_impl/main_pivoted /Data/tables/lineitem/comments.col %express%dependencies%
<src>/build/final_impl/main_pindex /Data/tables/lineitem/comments.col %express%dependencies%
```

The FFT implementation is in `fft-impl/batched-fft.cu`.

Example usage,
```
<src>/build/fft_impl/batched_fft /Data/tables/lineitem/comments.col dependencies
```


