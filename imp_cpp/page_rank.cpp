// On Linux and MacOS, you can compile and run this program like so:
//   g++ -std=c++11 -O3 -DNDEBUG -DTACO -I ../../include -L../../build/lib -ltaco spmv.cpp -o spmv
//   LD_LIBRARY_PATH=../../build/lib ./spmv

#include <taco.h>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include "cmd/cmd_handle.h"
#include "debug/utils_debug.h"
#include "algorithm/algo_interface.h"
#include "kernels/kernel_interface.h"
#include "utils/utils.h"

using namespace taco;

constexpr double PAGE_RANK_D = 0.85f;
constexpr double PAGE_RANK_MAX = 1.0f;
const char* MTX_DATA_PATH = "/Users/admin/workspace/page_rank/data/page_map.mtx";
// const char* MTX_DATA_PATH = "/Users/admin/workspace/page_rank/data/8x8-12.mtx";

int main(int argc, char* argv[]) {
    int ret_code = 0;
    CmdOpt cmd_opt = cmd_handle(argc, argv);

    /* init tensor */
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

    Tensor<double> y({A.getDimension(0)}, dv);

    taco_tensor_t* c_tensor_alpha = alpha.getTacoTensorT();
    taco_tensor_t* c_tensor_A = A.getTacoTensorT();
    taco_tensor_t* c_tensor_x = x.getTacoTensorT();
    taco_tensor_t* c_tensor_z = z.getTacoTensorT();
    taco_tensor_t* c_tensor_y = y.getTacoTensorT();

    FP_LOG(FP_LEVEL_INFO, "[assemble]\n");
    ret_code = assemble(c_tensor_y, c_tensor_alpha, c_tensor_A, c_tensor_x, c_tensor_z);
    ERROR_HANDLE_;

    taco_tensor_t* cur_result = nullptr;

    /* algorithm start */
    std::unordered_set<KernelType> kernels_set;
    kernels_set.insert(cmd_opt.kernel_type);
    auto algo_context = AlgoInterface::make(cmd_opt.algo_type, kernels_set);

    algo_context->upload(c_tensor_y, c_tensor_alpha, c_tensor_A, c_tensor_x, c_tensor_z);
    algo_context->run();
    algo_context->download(&cur_result);
    print_vector_tensor(cur_result);
}