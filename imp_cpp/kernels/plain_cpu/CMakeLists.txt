file(GLOB RAW_SOURCES
    "*.cpp"
)

set (OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I/usr/local/opt/libomp/include")
set (OpenMP_CXX_LIB_NAMES "omp")
set (OpenMP_omp_LIBRARY "/usr/local/opt/libomp/lib/libomp.dylib")

if(USE_OpenMP)
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} ")
endif(USE_OpenMP)

add_library(plain_cpu_kernel ${RAW_SOURCES})
target_link_libraries (plain_cpu_kernel tensor)

if(USE_OpenMP)
  target_link_libraries (plain_cpu_kernel ${OpenMP_CXX_LIB_NAMES})
endif(USE_OpenMP)
