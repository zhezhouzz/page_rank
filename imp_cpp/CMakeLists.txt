cmake_minimum_required (VERSION 3.5)
project (megface-origin-test)

# commandline option
SET(CUR_DEBUG_LEVEL "0" CACHE STRING "Build the project using debugging code")
option(USE_OpenMP "use openmp" OFF)
add_definitions(-DCUR_DEBUG_LEVEL=${CUR_DEBUG_LEVEL})

set (CMAKE_CXX_FLAGS "-std=c++14 ")

# perpare for sub CMakeLists.txt
include_directories ("${PROJECT_SOURCE_DIR}")

# start sub CMakeLists.txt
add_subdirectory (tensor)
add_subdirectory (cmd)
add_subdirectory (debug)
add_subdirectory (utils)
add_subdirectory (kernels)
add_subdirectory (algorithm)

# compile executable file
add_executable (page_rank.out page_rank.cpp)
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -framework OpenCL ")
target_link_libraries (page_rank.out algo kernels cmd utils utils_debug)

install (TARGETS page_rank.out DESTINATION /)