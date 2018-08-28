add_subdirectory (plain_cpu)
add_subdirectory (opencl)

file(GLOB RAW_SOURCES
    "*.cpp"
)

add_library(kernels ${RAW_SOURCES})
target_link_libraries (kernels plain_cpu_kernel opencl_kernel)