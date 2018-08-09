file(GLOB RAW_SOURCES
    "*.cpp"
)

add_library(opencl_kernel ${RAW_SOURCES})