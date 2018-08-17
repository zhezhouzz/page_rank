#include "utils.h"

int assemble(taco_tensor_t* y, taco_tensor_t* alpha, taco_tensor_t* A, taco_tensor_t* x,
             taco_tensor_t* z) {
    int y1_dimension = (int)(y->dimensions[y->mode_ordering[0]]);
    double* __restrict y_vals = (double*)(y->vals);
    int y_vals_size = y->vals_size;
    int* __restrict A1_pos = (int*)(A->indices[0][0]);
    int* __restrict A1_coord = (int*)(A->indices[0][1]);
    int* __restrict A2_pos = (int*)(A->indices[1][0]);
    int* __restrict A2_coord = (int*)(A->indices[1][1]);
    int z1_dimension = (int)(z->dimensions[z->mode_ordering[0]]);
    int32_t pA1 = A1_pos[0];
    int32_t A1_end = A1_pos[1];
    int32_t iz = 0;
    int32_t z1_end = z1_dimension;
    while (pA1 < A1_end) {
        int32_t iA = A1_coord[pA1];
        int32_t pz1 = iz;
        int32_t py1 = iz;
        if (iA == iz) {
            for (int32_t pA2 = A2_pos[pA1]; pA2 < A2_pos[(pA1 + 1)]; pA2++) {
                int32_t jA = A2_coord[pA2];
            }
        } else {
        }
        pA1 += (int32_t)(iA == iz);
        iz++;
    }
    while (iz < z1_end) {
        int32_t pz1 = iz;
        int32_t py1 = iz;
        iz++;
    }

    y_vals = (double*)malloc(sizeof(double) * y1_dimension);
    y_vals_size = y1_dimension;
    y->vals = (uint8_t*)y_vals;
    y->vals_size = y_vals_size;
    return 0;
}

void print_vector_tensor(taco_tensor_t* x, FpDebugLevel level) {
    FP_LOG(level, "tensor: [");
    for (int i = 0; i < (int)(x->dimensions[x->mode_ordering[0]]); i++) {
        FP_LOG(level, "%.10e ", ((double*)(x->vals))[i]);
    }
    FP_LOG(level, "]\n");
    return;
}

void print_vector_if_active(const std::vector<bool>& if_active) {
    FP_LOG(FP_LEVEL_INFO, "if_active: [");
    for(int i = 0 ; i < if_active.size(); i++) {
        if(if_active[i]) {
            FP_LOG(FP_LEVEL_INFO, "true, ");
        } else {
            FP_LOG(FP_LEVEL_INFO, "false, ");
        }
    }
    FP_LOG(FP_LEVEL_INFO, "]\n");
    return;
}