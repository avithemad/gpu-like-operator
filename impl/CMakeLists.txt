
add_executable(brute-force brute-force.cu)
add_executable(brute-force-pivoted brute-force-pivoted.cu)
add_executable(brute-force-bigram-index brute-force-bigram-index-non-split.cu)
# add_executable(brute-force-pivotedk brute-force-pivotedk.cu)
add_executable(kmp_basic kmp_basic.cu)
add_executable(sort-comments sort-comments.cu)
add_executable(fft fft.cu)



target_include_directories(brute-force PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../utils)
target_include_directories(brute-force-pivoted PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../utils)
target_include_directories(brute-force-bigram-index PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../utils)
# target_include_directories(brute-force-pivotedk PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../utils)
target_include_directories(kmp_basic PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../utils)
target_include_directories(sort-comments PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../utils)
target_include_directories(fft PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../utils)


target_link_libraries(fft PRIVATE "/usr/local/cuda-12/lib64/libcufft.so")
