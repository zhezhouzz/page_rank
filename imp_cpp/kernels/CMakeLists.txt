add_subdirectory (taco)
add_subdirectory (opencl)

file(GLOB RAW_SOURCES
    "*.cpp"
)

add_library(kernels ${RAW_SOURCES})
target_link_libraries (kernels taco_kernel opencl_kernel)