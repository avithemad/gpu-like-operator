
add_executable(brute-force brute-force.cu)



target_include_directories(brute-force PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../utils)
