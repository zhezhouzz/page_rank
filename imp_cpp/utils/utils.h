#pragma once

#include <taco.h>

#define ERROR_HANDLE_                                                                  \
    do {                                                                               \
        if (ret_code != 0) {                                                           \
            std::cout << "[line " << __LINE__ << "] error: " << ret_code << std::endl; \
            return 0;                                                                  \
        }                                                                              \
    } while (0)

int assemble(taco_tensor_t* y, taco_tensor_t* alpha, taco_tensor_t* A, taco_tensor_t* x,
             taco_tensor_t* z);

void print_vector_tensor(taco_tensor_t* x);