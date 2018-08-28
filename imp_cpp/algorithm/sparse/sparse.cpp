#include <unordered_map>
#include "sparse.h"
#include "algorithm/algo_interface.h"
#include "debug/utils_debug.h"
#include "kernels/kernel_interface.h"
#include "utils/utils.h"

AlgoSparse::AlgoSparse(std::unordered_set<KernelType> needed_kernels) {
    assert(needed_kernels.size() > 0);
    for (const auto& kernel_type : needed_kernels) {
        kernels_hashmap[kernel_type] = KernelInterface::make(kernel_type);
    }
    return;
}

int AlgoSparse::upload(std::shared_ptr<Tensor> y, std::shared_ptr<Tensor> alpha, std::shared_ptr<Tensor> A,
                          std::shared_ptr<Tensor> x, std::shared_ptr<Tensor> z, const CmdOpt& option) {
    auto default_kernel = kernels_hashmap.begin()->second;
    default_kernel->upload(y, alpha, A, x, z);
    _eps = option.eps;
    return 0;
}

int AlgoSparse::run() {
    int ret_code = 0;
    bool flag_x2y = true;
    auto default_kernel = kernels_hashmap.begin()->second;

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
                ret_code =
                    default_kernel->page_rank_once(flag_x2y
                        );
                ERROR_HANDLE_;
                ret_code = default_kernel->download(flag_x2y, pre_result, cur_result);
            }

            FP_LOG(FP_LEVEL_INFO, "[vetor_norm]\n");
            {
                FPDebugTimer timer_norm(FP_LEVEL_INFO, __FILE__, __LINE__);
                norm = default_kernel->vetor_norm(pre_result, cur_result);
            }
            FP_LOG(FP_LEVEL_INFO, "norm = %.10e\n", norm);
            flag_x2y = not flag_x2y;
#ifndef FPOPT
            times++;
            FP_LOG(FP_LEVEL_INFO, "<loop %d>\n", times);
            cur_result->print();
#endif
        } while (norm > _eps);
    }
#ifndef FPOPT
    FP_LOG(FP_LEVEL_ERROR, "loop: %d times\n", times);
#endif
    return 0;
}

int AlgoSparse::download(std::shared_ptr<Tensor>& result) const
{
    result = cur_result;
    return 0;
}
