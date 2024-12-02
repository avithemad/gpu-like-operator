
add_executable(main main.cu)



target_include_directories(main PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../utils ${CMAKE_CURRENT_SOURCE_DIR})


target_link_libraries(main PRIVATE "/usr/local/cuda-12/lib64/libcufft.so")
