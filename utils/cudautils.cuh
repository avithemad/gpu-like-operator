#pragma once
#include <iostream>
void CUDACHKERR() {

  auto err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cout << "CUDA ERROR: " << cudaGetErrorString(err) << "\n"; 
  }
} 