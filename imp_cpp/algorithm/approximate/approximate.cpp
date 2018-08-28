#include "approximate.h"
#include <unordered_map>
#include "algorithm/algo_interface.h"
#include "debug/utils_debug.h"
#include "kernels/kernel_interface.h"
#include "utils/utils.h"

AlgoApproximate::AlgoApproximate(std::unordered_set<KernelType> needed_kernels) {
    assert(needed_kernels.size() > 0);
    for (const auto& kernel_type : needed_kernels) {
        kernels_hashmap[kernel_type] = KernelInterface::make(kernel_type);
    }
    return;
}

int AlgoApproximate::upload(std::shared_ptr<Tensor> y, std::shared_ptr<Tensor> alpha, std::shared_ptr<Tensor> A,
                            std::shared_ptr<Tensor> x, std::shared_ptr<Tensor> z, const CmdOpt& option) {
    auto default_kernel = kernels_hashmap.begin()->second;
    default_kernel->upload_approximate_mxv(y, alpha, A, x, z);
    if_active.resize((int)(x->dimensions[0]), true);
    _eps = option.eps;
    _inactive_tolerance = option.inactive_tolerance;
    _terminate_active_rate = option.terminate_active_rate;
    _terminate_min =
        static_cast<int>((int)(x->dimensions[0]) * _terminate_active_rate);
    return 0;
}

int AlgoApproximate::run() {
    int ret_code = 0;
    bool flag_x2y = true;
    auto default_kernel = kernels_hashmap.begin()->second;
    int active_num = 0;

#ifndef FPOPT
    int times = 0;
#endif
    {
        FPDebugTimer timer_page_rank(FP_LEVEL_WARNING, __FILE__, __LINE__);
        double norm = std::numeric_limits<double>::max();
        do {
            FP_LOG(FP_LEVEL_INFO, "[normalize]\n");
            {
                FPDebugTimer timer_compute(FP_LEVEL_INFO, __FILE__, __LINE__);
                ret_code = default_kernel->normalize(flag_x2y, if_active);
                ERROR_HANDLE_;
            }

            FP_LOG(FP_LEVEL_INFO, "[approximate]\n");
            {
                FPDebugTimer timer_compute(FP_LEVEL_INFO, __FILE__, __LINE__);
                ret_code = default_kernel->approximate_mxv(flag_x2y, if_active);
                ERROR_HANDLE_;
                ret_code = default_kernel->download(flag_x2y, pre_result, cur_result);
            }

            FP_LOG(FP_LEVEL_INFO, "[find_active]\n");
            {
                FPDebugTimer timer_norm(FP_LEVEL_INFO, __FILE__, __LINE__);
                ret_code = default_kernel->approximate_find_active(
                    pre_result, cur_result, if_active, _eps, _inactive_tolerance);
            }

            active_num = 0;
            for (const auto& active : if_active) {
                if (active) {
                    active_num++;
                }
            }
            FP_LOG(FP_LEVEL_INFO, "active_num = %d\n", int(active_num));
            flag_x2y = not flag_x2y;
#ifndef FPOPT
            times++;
            FP_LOG(FP_LEVEL_INFO, "<loop %d>\n", times);
            // pre_result->print();
            // cur_result->print();
#endif
        } while (active_num > _terminate_min);
    }
#ifndef FPOPT
    FP_LOG(FP_LEVEL_ERROR, "loop: %d times\n", times);
#endif
    return 0;
}

int AlgoApproximate::download(std::shared_ptr<Tensor>& result) const {
    result = cur_result;
    return 0;
}
