file(GLOB RAW_SOURCES
    "*.cpp"
)

# set (DOpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I/usr/local/opt/libomp/include")
# set (DOpenMP_CXX_LIB_NAMES "omp")
# set (OpenMP_omp_LIBRARY "/usr/local/opt/libomp/lib/libomp.dylib")

add_library(taco_kernel ${RAW_SOURCES})
target_link_libraries (taco_kernel taco)