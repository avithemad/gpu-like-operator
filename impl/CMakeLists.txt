
add_executable(brute-force brute-force.cu)
add_executable(brute-force-pivoted brute-force-pivoted.cu)



target_include_directories(brute-force PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../utils)
target_include_directories(brute-force-pivoted PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../utils)
