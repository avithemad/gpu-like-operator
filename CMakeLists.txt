cmake_minimum_required(VERSION 3.28.3)

project(GPULIKE VERSION 0.0.1 LANGUAGES CXX CUDA)


set(ARROW_DIR "" CACHE PATH "Arrow installation directory")

# 1.1 Check the "include" directory
set(ARROW_INCLUDE_DIR "${ARROW_DIR}/include")
if(NOT EXISTS "${ARROW_INCLUDE_DIR}")
message(FATAL_ERROR
  " ARROW_DIR (${ARROW_INCLUDE_DIR}) is invalid.")
endif()

set(ARROW_LIB_DIR "${ARROW_DIR}/lib")
if(NOT EXISTS "${ARROW_LIB_DIR}")
message(FATAL_ERROR
  " ARROW_DIR (${ARROW_LIB_DIR}) is invalid.")
endif()


add_subdirectory(impl)