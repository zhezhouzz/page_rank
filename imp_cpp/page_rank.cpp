// On Linux and MacOS, you can compile and run this program like so:
//   g++ -std=c++11 -O3 -DNDEBUG -DTACO -I ../../include -L../../build/lib -ltaco spmv.cpp -o spmv
//   LD_LIBRARY_PATH=../../build/lib ./spmv

#include <taco.h>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include "cmd_handle.h"
#include "debug/utils_debug.h"
#include "kernels/opencl/clhost.h"
#include "kernels/taco/taco_page_rank.h"

using namespace taco;

#define ERROR_HANDLE_                                                                  \
    do {                                                                               \
        if (ret_code != 0) {                                                           \
            std::cout << "[line " << __LINE__ << "] error: " << ret_code << std::endl; \
        }                                                                              \
    } while (0)

constexpr double PAGE_RANK_EPS = 1.0e-8;
constexpr double PAGE_RANK_D = 0.85f;
constexpr double PAGE_RANK_MAX = 1.0f;
const char* MTX_DATA_PATH = "/Users/admin/workspace/page_rank/data/page_map.mtx";
// const char* MTX_DATA_PATH = "/Users/admin/workspace/page_rank/data/8x8-12.mtx";

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

void print_vector_tensor(taco_tensor_t* x) {
    FP_LOG(FP_LEVEL_INFO, "vector: [");
    for (int i = 0; i < (int)(x->dimensions[x->mode_ordering[0]]); i++) {
        FP_LOG(FP_LEVEL_INFO, "%.10e ", ((double*)(x->vals))[i]);
    }
    FP_LOG(FP_LEVEL_INFO, "]\n");
    return;
}

int main(int argc, char* argv[]) {
    CmdOpt cmd_opt = cmd_handle(argc, argv);
    std::default_random_engine gen(0);
    std::uniform_real_distribution<double> unif(0.0, 1.0);

    Format csr({Sparse, Sparse});
    Format dv({Dense});

    Tensor<double> A = read(MTX_DATA_PATH, csr);
    FP_LOG(FP_LEVEL_INFO, "[LOAD FINISHED]\n");

    Tensor<double> x({A.getDimension(1)}, dv);
    int length = x.getDimension(0);
    for (int i = 0; i < length; ++i) {
        x.insert({i}, (double)(PAGE_RANK_MAX / length));
    }
    x.pack();

    Tensor<double> alpha(PAGE_RANK_D);

    Tensor<double> z({A.getDimension(0)}, dv);
    for (int i = 0; i < z.getDimension(0); ++i) {
        z.insert({i}, (double)((PAGE_RANK_MAX - PAGE_RANK_D) / length));
    }
    z.pack();

    taco_tensor_t* c_tensor_alpha = alpha.getTacoTensorT();
    taco_tensor_t* c_tensor_A = A.getTacoTensorT();
    taco_tensor_t* c_tensor_x = x.getTacoTensorT();
    taco_tensor_t* c_tensor_z = z.getTacoTensorT();

    Tensor<double> y({A.getDimension(0)}, dv);
    taco_tensor_t* c_tensor_y = y.getTacoTensorT();
    int ret_code = 0;
    FP_LOG(FP_LEVEL_INFO, "[assemble]\n");
    ret_code = assemble(c_tensor_y, c_tensor_alpha, c_tensor_A, c_tensor_x, c_tensor_z);
    ERROR_HANDLE_;

    prepare(cmd_opt.kernel_type);
    upload(c_tensor_y, c_tensor_alpha, c_tensor_A, c_tensor_x, c_tensor_z, cmd_opt.kernel_type);

    taco_tensor_t* pre_result = nullptr;
    taco_tensor_t* cur_result = nullptr;
#ifndef FPOPT
    int times = 0;
#endif
    {
        FPDebugTimer timer_page_rank(FP_LEVEL_WARNING, __FILE__, __LINE__);
        double norm = std::numeric_limits<double>::max();
        do {
            FP_LOG(FP_LEVEL_INFO, "[page_rank]\n");
            {
                FPDebugTimer timer_compute(FP_LEVEL_INFO, __FILE__, __LINE__);
                ret_code = page_rank_once(c_tensor_y, c_tensor_alpha, c_tensor_A, c_tensor_x,
                                          c_tensor_z, cmd_opt.kernel_type);
                ERROR_HANDLE_;
                ret_code = download(&pre_result, &cur_result, cmd_opt.kernel_type);
            }

            FP_LOG(FP_LEVEL_INFO, "[vetor_norm]\n");
            {
                FPDebugTimer timer_norm(FP_LEVEL_INFO, __FILE__, __LINE__);
                norm = vetor_norm(pre_result, cur_result, cmd_opt.kernel_type);
            }
            FP_LOG(FP_LEVEL_INFO, "norm = %.10e\n", norm);
#ifndef FPOPT
            times++;
            FP_LOG(FP_LEVEL_INFO, "<loop %d>\n", times);
            print_vector_tensor(cur_result);
#endif
        } while (norm > PAGE_RANK_EPS);
    }
    finish(cmd_opt.kernel_type);
#ifndef FPOPT
    FP_LOG(FP_LEVEL_WARNING, "loop: %d times\n", times);
#endif
    print_vector_tensor(cur_result);
}