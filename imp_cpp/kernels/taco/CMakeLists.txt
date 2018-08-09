file(GLOB RAW_SOURCES
    "*.cpp"
)

add_library(taco_kernel ${RAW_SOURCES})
target_link_libraries (taco_kernel taco)