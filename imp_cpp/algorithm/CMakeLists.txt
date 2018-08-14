add_subdirectory (sparse)

file(GLOB RAW_SOURCES
    "*.cpp"
)

add_library(algo ${RAW_SOURCES})
target_link_libraries (algo algo_sparse)