add_subdirectory (sparse)
add_subdirectory (dense)
add_subdirectory (approximate)

file(GLOB RAW_SOURCES
    "*.cpp"
)

add_library(algo ${RAW_SOURCES})
target_link_libraries (algo algo_sparse algo_dense algo_approximate)