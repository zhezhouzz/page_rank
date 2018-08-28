#include "cpu_page_rank.h"
#include <cstdlib>
#include "debug/utils_debug.h"
#include "utils/utils.h"
#include <cmath>

static int cpu_page_rank_(std::shared_ptr<Tensor> y, std::shared_ptr<Tensor> alpha, std::shared_ptr<Tensor> A,
                           std::shared_ptr<Tensor> x, std::shared_ptr<Tensor> z) {
    int y1_dimension = (int)(y->dimensions[0]);
    double* __restrict y_vals = (double*)(y->vals);
    double* __restrict alpha_vals = (double*)(alpha->vals);
    double* __restrict A_vals = (double*)(A->vals);
    int A1_dimension = (int)(A->dimensions[0]);
    uint32_t* __restrict A2_pos = (uint32_t*)(A->indices.data());
    uint32_t* __restrict A2_coord = (uint32_t*)(A->cols);
    double* __restrict x_vals = (double*)(x->vals);
    int z1_dimension = (int)(z->dimensions[0]);
    double* __restrict z_vals = (double*)(z->vals);
    for (int32_t py = 0; py < y1_dimension; py++) {
        y_vals[py] = 0;
    }
    int32_t pA1 = 0;
    int32_t A1_end = A1_dimension;
    int32_t iz = 0;
    int32_t z1_end = z1_dimension;
    int32_t pA2 = 0;
    while (pA1 < A1_end) {
        int32_t iA = pA1;
        int32_t py1 = iz;
        int32_t pz1 = iz;
        if (iA == iz) {
            double tj = 0;
            for (; pA2 < A2_pos[pA1]; pA2++) {
                int32_t jA = A2_coord[pA2];
                tj += A_vals[pA2] * x_vals[jA];
                FP_LOG(FP_LEVEL_INFO, "  %fx%f=%f\n", A_vals[pA2], x_vals[jA],
                       A_vals[pA2] * x_vals[jA]);
            }
            y_vals[py1] = alpha_vals[0] * tj + z_vals[pz1];
            FP_LOG(FP_LEVEL_INFO, "%fx%f + %f=%f\n", alpha_vals[0], tj, z_vals[pz1],
            y_vals[py1]);
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

static int dense_mxv_(std::shared_ptr<Tensor> y, std::shared_ptr<Tensor> alpha, std::shared_ptr<Tensor> A, std::shared_ptr<Tensor> x,
                      std::shared_ptr<Tensor> z) {
    double* __restrict y_vals = (double*)(y->vals);
    double* __restrict alpha_vals = (double*)(alpha->vals);
    int A1_dimension = (int)(A->dimensions[0]);
    int A2_dimension = (int)(A->dimensions[1]);
    double* __restrict A_vals = (double*)(A->vals);
    double* __restrict x_vals = (double*)(x->vals);
    double* __restrict z_vals = (double*)(z->vals);
#pragma omp parallel for
    for (int32_t iA = 0; iA < A1_dimension; iA++) {
        double tj = 0;
        for (int32_t jA = 0; jA < A2_dimension; jA++) {
            int32_t pA2 = iA * A2_dimension + jA;
            tj += A_vals[pA2] * x_vals[jA];
            // FP_LOG(FP_LEVEL_INFO, "  %fx%f=%f\n", A_vals[pA2], x_vals[jA],
            //        A_vals[pA2] * x_vals[jA]);
        }
        y_vals[iA] = alpha_vals[0] * tj + z_vals[iA];
        // FP_LOG(FP_LEVEL_INFO, "%fx%f + %f=%f\n", alpha_vals[0], tj, z_vals[iA], y_vals[iA]);
    }
    return 0;
}

static int approximate_mxv_(std::shared_ptr<Tensor> y, std::shared_ptr<Tensor> A, std::shared_ptr<Tensor> x,
                            std::vector<bool> if_active) {
    double* __restrict y_vals = (double*)(y->vals);
    int A1_dimension = (int)(A->dimensions[0]);
    int A2_dimension = (int)(A->dimensions[1]);
    double* __restrict A_vals = (double*)(A->vals);
    double* __restrict x_vals = (double*)(x->vals);
#pragma omp parallel for
    for (int32_t iA = 0; iA < A1_dimension; iA++) {
        double tj = 0;
        if (if_active[iA]) {
            for (int32_t jA = 0; jA < A2_dimension; jA++) {
                int32_t pA2 = iA * A2_dimension + jA;
                tj += A_vals[pA2] * x_vals[jA];
            }
        } else {
            tj = x_vals[iA];
        }
        y_vals[iA] = tj;
    }
    return 0;
}

int KernelCpu::upload(std::shared_ptr<Tensor> y, std::shared_ptr<Tensor> alpha, std::shared_ptr<Tensor> A, std::shared_ptr<Tensor> x,
                       std::shared_ptr<Tensor> z) {
    _y = y;
    _alpha = alpha;
    _A = A;
    _x = x;
    _z = z;
    return 0;
}

int KernelCpu::page_rank_once(bool flag_x2y) {
    if (flag_x2y) {
        FP_LOG(FP_LEVEL_INFO, "flag_x2y = true\n");
        return cpu_page_rank_(_y, _alpha, _A, _x, _z);
    } else {
        return cpu_page_rank_(_x, _alpha, _A, _y, _z);
    }
}

int KernelCpu::upload_dense_mxv(std::shared_ptr<Tensor> y, std::shared_ptr<Tensor> alpha, std::shared_ptr<Tensor> A,
                                 std::shared_ptr<Tensor> x, std::shared_ptr<Tensor> z) {
    _y = y;
    _alpha = alpha;
    _A = A;
    _x = x;
    _z = z;
    return 0;
}

int KernelCpu::dense_mxv(bool flag_x2y) {
    if (flag_x2y) {
        FP_LOG(FP_LEVEL_INFO, "flag_x2y = true\n");
        return dense_mxv_(_y, _alpha, _A, _x, _z);
    } else {
        return dense_mxv_(_x, _alpha, _A, _y, _z);
    }
}

int KernelCpu::upload_approximate_mxv(std::shared_ptr<Tensor> y, std::shared_ptr<Tensor> alpha, std::shared_ptr<Tensor> A,
                                       std::shared_ptr<Tensor> x, std::shared_ptr<Tensor> z) {
    _y = y;
    _alpha = alpha;
    _A = A;
    _x = x;
    _z = z;
    double* __restrict alpha_vals = (double*)(alpha->vals);
    int A1_dimension = (int)(A->dimensions[0]);
    int A2_dimension = (int)(A->dimensions[1]);
    double* __restrict A_vals = (double*)(A->vals);
    double* __restrict z_vals = (double*)(z->vals);

    double remain_factor = (1 - alpha_vals[0]) / A1_dimension;
    double outflow_factor = alpha_vals[0];
#pragma omp parallel for
    for (int32_t iA = 0; iA < A1_dimension; iA++) {
        double tj = 0;
        for (int32_t jA = 0; jA < A2_dimension; jA++) {
            int32_t pA2 = iA * A2_dimension + jA;
            A_vals[pA2] = A_vals[pA2] * outflow_factor + remain_factor;
            // FP_LOG(FP_LEVEL_INFO, "  A_vals[%d]=%f\n", pA2, A_vals[pA2]);
        }
    }

    _history_active_table.resize(A1_dimension, 0);
    return 0;
}

int KernelCpu::approximate_mxv(bool flag_x2y, std::vector<bool> if_active) {
    print_vector_if_active(if_active);
    if (flag_x2y) {
        FP_LOG(FP_LEVEL_INFO, "{x --> y}\n");
        return approximate_mxv_(_y, _A, _x, if_active);
    } else {
        FP_LOG(FP_LEVEL_INFO, "{x <-- y}\n");
        return approximate_mxv_(_x, _A, _y, if_active);
    }
}

int KernelCpu::approximate_find_active(std::shared_ptr<Tensor> x, std::shared_ptr<Tensor> y,
                                        std::vector<bool>& if_active, double eps, int stable_num) {
    int x1_dimension = (int)(x->dimensions[0]);
    assert(x1_dimension == if_active.size());
    double* __restrict x_vals = (double*)(x->vals);
    double* __restrict y_vals = (double*)(y->vals);
    // TODO optimate this, save the calculation
    for (int i = 0; i < x1_dimension; i++) {
        if (std::fabs(x_vals[i] - y_vals[i]) / x_vals[i] < eps) {
            _history_active_table[i]++;
        }
        if (_history_active_table[i] >= stable_num) {
            if_active[i] = false;
        } else {
            if_active[i] = true;
        }
    }
    return 0;
}

int KernelCpu::normalize(bool flag_x2y, std::vector<bool>& if_active) {
    if (flag_x2y) {
        int y1_dimension = (int)(_y->dimensions[0]);
        assert(y1_dimension == if_active.size());
        double* __restrict y_vals = (double*)(_y->vals);
        double total_active_flow = 0;
        double total_inactive_flow = 0;
        // TODO optimate this, save the calculation
        for (int i = 0; i < y1_dimension; i++) {
            if (if_active[i]) {
                total_active_flow += y_vals[i];
            } else {
                total_inactive_flow += y_vals[i];
            }
        }
        double inactive_factor = (1 - total_active_flow) / total_inactive_flow;
        for (int i = 0; i < y1_dimension; i++) {
            if (not if_active[i]) {
                y_vals[i] = y_vals[i] * inactive_factor;
            }
        }
    } else {
        int x1_dimension = (int)(_x->dimensions[0]);
        assert(x1_dimension == if_active.size());
        double* __restrict x_vals = (double*)(_x->vals);
        double total_active_flow = 0;
        double total_inactive_flow = 0;
        // TODO optimate this, save the calculation
        for (int i = 0; i < x1_dimension; i++) {
            if (if_active[i]) {
                total_active_flow += x_vals[i];
            } else {
                total_inactive_flow += x_vals[i];
            }
        }
        double inactive_factor = (1 - total_active_flow) / total_inactive_flow;
        for (int i = 0; i < x1_dimension; i++) {
            if (not if_active[i]) {
                x_vals[i] = x_vals[i] * inactive_factor;
            }
        }
    }
    return 0;
}

int KernelCpu::download(bool flag_x2y, std::shared_ptr<Tensor>& x, std::shared_ptr<Tensor>& y) {
    if (flag_x2y) {
        x = _y;
        y = _x;
    } else {
        x = _x;
        y = _y;
    }
    return 0;
}

int swap_vector(std::shared_ptr<Tensor> x, std::shared_ptr<Tensor> y) {
    double* __restrict x_vals = (double*)(x->vals);
    int y1_dimension = (int)(y->dimensions[0]);
    double* __restrict y_vals = (double*)(y->vals);
#pragma omp parallel for
    for (int32_t iy = 0; iy < y1_dimension; iy++) {
        x_vals[iy] = y_vals[iy];
    }
    return 0;
}

double KernelCpu::vetor_norm(std::shared_ptr<Tensor> x, std::shared_ptr<Tensor> y) const {
    int y1_dimension = (int)(y->dimensions[0]);
    int x1_dimension = (int)(x->dimensions[0]);
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