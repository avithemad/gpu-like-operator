
add_executable(main main.cu)
add_executable(main_pivoted main_pivoted.cu)
add_executable(main_pindex main_pindex.cu)



target_include_directories(main PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../utils ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(main_pivoted PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../utils ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(main_pindex PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../utils ${CMAKE_CURRENT_SOURCE_DIR})


