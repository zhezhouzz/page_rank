#include "taco_page_rank.h"
#include "debug/utils_debug.h"

namespace {
taco_tensor_t* y;
taco_tensor_t* alpha;
taco_tensor_t* A;
taco_tensor_t* x;
taco_tensor_t* z;
};  // namespace

int taco_page_rank_(taco_tensor_t* y, taco_tensor_t* alpha, taco_tensor_t* A, taco_tensor_t* x,
                    taco_tensor_t* z) {
    int y1_dimension = (int)(y->dimensions[y->mode_ordering[0]]);
    double* __restrict y_vals = (double*)(y->vals);
    double* __restrict alpha_vals = (double*)(alpha->vals);
    int* __restrict A1_pos = (int*)(A->indices[0][0]);
    int* __restrict A1_coord = (int*)(A->indices[0][1]);
    int* __restrict A2_pos = (int*)(A->indices[1][0]);
    int* __restrict A2_coord = (int*)(A->indices[1][1]);
    double* __restrict A_vals = (double*)(A->vals);
    double* __restrict x_vals = (double*)(x->vals);
    int z1_dimension = (int)(z->dimensions[z->mode_ordering[0]]);
    double* __restrict z_vals = (double*)(z->vals);
    for (int32_t py = 0; py < y1_dimension; py++) {
        y_vals[py] = 0;
    }
    int32_t pA1 = A1_pos[0];
    int32_t A1_end = A1_pos[1];
    int32_t iz = 0;
    int32_t z1_end = z1_dimension;
    while (pA1 < A1_end) {
        int32_t iA = A1_coord[pA1];
        int32_t pz1 = iz;
        int32_t py1 = iz;
        if (iA == iz) {
            double tj = 0;
            for (int32_t pA2 = A2_pos[pA1]; pA2 < A2_pos[(pA1 + 1)]; pA2++) {
                int32_t jA = A2_coord[pA2];
                tj += A_vals[pA2] * x_vals[jA];
                FP_LOG(FP_LEVEL_INFO, "  %fx%f=%f\n", A_vals[pA2], x_vals[jA],
                       A_vals[pA2] * x_vals[jA]);
            }
            y_vals[py1] = alpha_vals[0] * tj + z_vals[pz1];
            FP_LOG(FP_LEVEL_INFO, "%fx%f + %f=%f\n", alpha_vals[0], tj, z_vals[pz1], y_vals[py1]);
        } else {
            y_vals[py1] = z_vals[pz1];
        }
        pA1 += (int32_t)(iA == iz);
        iz++;
    }
    while (iz < z1_end) {
        int32_t pz1 = iz;
        int32_t py1 = iz;
        y_vals[py1] = z_vals[pz1];
        iz++;
    }
    return 0;
}

int taco_upload(taco_tensor_t* y, taco_tensor_t* alpha, taco_tensor_t* A, taco_tensor_t* x,
                taco_tensor_t* z) {
    ::y = y;
    ::alpha = alpha;
    ::A = A;
    ::x = x;
    ::z = z;
    return 0;
}

int taco_page_rank(bool flag_x2y) {
    if (flag_x2y) {
        return taco_page_rank_(::y, ::alpha, ::A, ::x, ::z);
    } else {
        return taco_page_rank_(::x, ::alpha, ::A, ::y, ::z);
    }
}

int taco_download(bool flag_x2y, taco_tensor_t** x, taco_tensor_t** y) {
    if (flag_x2y) {
        *x = ::y;
        *y = ::x;
    } else {
        *x = ::x;
        *y = ::y;
    }
    return 0;
}

int taco_swap_vector(taco_tensor_t* x, taco_tensor_t* y) {
    double* __restrict x_vals = (double*)(x->vals);
    int y1_dimension = (int)(y->dimensions[y->mode_ordering[0]]);
    double* __restrict y_vals = (double*)(y->vals);
#pragma omp parallel for
    for (int32_t iy = 0; iy < y1_dimension; iy++) {
        x_vals[iy] = y_vals[iy];
    }
    return 0;
}

double taco_vetor_norm(taco_tensor_t* x, taco_tensor_t* y) {
    int y1_dimension = (int)(y->dimensions[y->mode_ordering[0]]);
    int x1_dimension = (int)(x->dimensions[x->mode_ordering[0]]);
    assert(y1_dimension == x1_dimension);
    double* __restrict x_vals = (double*)(x->vals);
    double* __restrict y_vals = (double*)(y->vals);
    double norm = 0;
#pragma omp parallel for
    for (int32_t iy = 0; iy < y1_dimension; iy++) {
        double diff = x_vals[iy] - y_vals[iy];
        norm += diff * diff;
    }
    return norm;
}