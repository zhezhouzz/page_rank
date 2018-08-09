cmake_minimum_required (VERSION 3.5)
project (megface-origin-test)

# commandline option
option (USE_OPENCL "Build the project using debugging code" OFF)
if (USE_OPENCL)
	add_definitions(-DUSE_OPENCL=ON)
endif()
SET(CUR_DEBUG_LEVEL "0" CACHE STRING "Build the project using debugging code")
add_definitions(-DCUR_DEBUG_LEVEL=${CUR_DEBUG_LEVEL})

set (CMAKE_CXX_FLAGS "-std=c++14 ")

# perpare for sub CMakeLists.txt
include_directories ("${PROJECT_SOURCE_DIR}")

# start sub CMakeLists.txt
add_subdirectory (debug)
add_subdirectory (kernels)

# compile executable file
add_executable (page_rank.out page_rank.cpp cmd_handle.cpp)
if (USE_OPENCL)
	set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -framework OpenCL ")
	target_link_libraries (page_rank.out kernels utils_debug)
else ()
	target_link_libraries (page_rank.out kernels utils_debug)
endif()

install (TARGETS page_rank.out DESTINATION /)