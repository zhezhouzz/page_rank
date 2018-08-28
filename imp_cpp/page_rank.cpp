#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include "algorithm/algo_interface.h"
#include "cmd/cmd_handle.h"
#include "debug/utils_debug.h"
#include "default_config.h"
#include "kernels/kernel_interface.h"
#include "tensor/tensor.h"
#include "utils/utils.h"

int main(int argc, char* argv[]) {
    int ret_code = 0;
    CmdOpt cmd_opt = cmd_handle(argc, argv);

    /* init tensor */
    std::shared_ptr<Tensor> c_tensor_alpha = std::make_shared<Tensor>(PAGE_RANK_D);
    std::shared_ptr<Tensor> c_tensor_A = std::make_shared<Tensor>(
        (cmd_opt.algo_type == AlgoType::sparse) ? TENSOR_MODE::TENSOR_MODE_SPARSE
                                                  : TENSOR_MODE::TENSOR_MODE_DENSE,
        cmd_opt.data_set_path);
    FP_LOG(FP_LEVEL_INFO, "[LOAD FINISHED]\n");

    int length = c_tensor_A->dimensions[0];
    std::shared_ptr<Tensor> c_tensor_x =
        std::make_shared<Tensor>(TENSOR_MODE::TENSOR_MODE_DENSE, std::vector<int>(1, length));
    for (int i = 0; i < length; ++i) {
        reinterpret_cast<double*>(c_tensor_x->vals)[i] = (double)(PAGE_RANK_MAX / length);
    }
    std::shared_ptr<Tensor> c_tensor_z =
        std::make_shared<Tensor>(TENSOR_MODE::TENSOR_MODE_DENSE, std::vector<int>(1, length));
    for (int i = 0; i < length; ++i) {
        reinterpret_cast<double*>(c_tensor_z->vals)[i] =
            (double)((PAGE_RANK_MAX - PAGE_RANK_D) / length);
    }
    std::shared_ptr<Tensor> c_tensor_y =
        std::make_shared<Tensor>(TENSOR_MODE::TENSOR_MODE_DENSE, std::vector<int>(1, length));

    c_tensor_x->print();
    c_tensor_z->print();

    std::shared_ptr<Tensor> cur_result = nullptr;

    /* algorithm start */
    std::unordered_set<KernelType> kernels_set;
    kernels_set.insert(cmd_opt.kernel_type);
    auto algo_context = AlgoInterface::make(cmd_opt.algo_type, kernels_set);
    algo_context->upload(c_tensor_y, c_tensor_alpha, c_tensor_A, c_tensor_x, c_tensor_z, cmd_opt);
    {
        FPDebugTimer timer_compute(FP_LEVEL_ERROR, "final-compute", 0);
        algo_context->run();
        algo_context->download(cur_result);
    }
    cur_result->print(FP_LEVEL_ERROR);
    cur_result->save("result.vector");
    return 0;
}