#pragma once
#include "debug/utils_debug.h"
#include <iostream>
#include <vector>

#define ERROR_HANDLE_                                                                  \
    do {                                                                               \
        if (ret_code != 0) {                                                           \
            std::cout << "[line " << __LINE__ << "] error: " << ret_code << std::endl; \
            return 0;                                                                  \
        }                                                                              \
    } while (0)

void print_vector_if_active(const std::vector<bool>& if_active);