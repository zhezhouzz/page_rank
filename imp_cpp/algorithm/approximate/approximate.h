#pragma once
#include <unordered_map>
#include "algorithm/algo_interface.h"

class AlgoApproximate final : public AlgoInterface {
public:
    AlgoApproximate(std::unordered_set<KernelType> needed_kernels);
    ~AlgoApproximate() = default;
    int upload(taco_tensor_t* y, taco_tensor_t* alpha, taco_tensor_t* A, taco_tensor_t* x,
               taco_tensor_t* z, const CmdOpt& option) override;
    int run() override;
    int download(taco_tensor_t** result) const override;
    std::unordered_map<KernelType, std::shared_ptr<KernelInterface>> kernels_hashmap;
    taco_tensor_t* pre_result = nullptr;
    taco_tensor_t* cur_result = nullptr;
    std::vector<bool> if_active;
    double _eps;
    int _inactive_tolerance;
    double _terminate_active_rate;
    int _terminate_min;
};