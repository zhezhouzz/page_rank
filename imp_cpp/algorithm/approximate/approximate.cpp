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

int AlgoApproximate::upload(taco_tensor_t* y, taco_tensor_t* alpha, taco_tensor_t* A,
                            taco_tensor_t* x, taco_tensor_t* z) {
    auto default_kernel = kernels_hashmap.begin()->second;
    default_kernel->upload_approximate_mxv(y, alpha, A, x, z);
    if_active.resize(5, true);
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
                ret_code = default_kernel->download(flag_x2y, &pre_result, &cur_result);
            }

            FP_LOG(FP_LEVEL_INFO, "[find_active]\n");
            {
                FPDebugTimer timer_norm(FP_LEVEL_INFO, __FILE__, __LINE__);
                ret_code = default_kernel->approximate_find_active(pre_result, cur_result,
                                                                   if_active, 0.0001, 3);
            }

            active_num = 0;
            for(const auto& active:if_active) {
                if(active) {
                    active_num++;
                }
            }
            FP_LOG(FP_LEVEL_INFO, "active_num = %d\n", int(active_num));
            flag_x2y = not flag_x2y;
#ifndef FPOPT
            times++;
            FP_LOG(FP_LEVEL_INFO, "<loop %d>\n", times);
            print_vector_tensor(pre_result);
            print_vector_tensor(cur_result);
#endif
        } while (active_num > 0);
    }
#ifndef FPOPT
    FP_LOG(FP_LEVEL_ERROR, "loop: %d times\n", times);
#endif
    return 0;
}

int AlgoApproximate::download(taco_tensor_t** result) const {
    *result = cur_result;
    return 0;
}
