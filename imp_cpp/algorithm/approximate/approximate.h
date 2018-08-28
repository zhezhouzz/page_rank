#pragma once
#include <unordered_map>
#include "algorithm/algo_interface.h"

class AlgoApproximate final : public AlgoInterface {
public:
    AlgoApproximate(std::unordered_set<KernelType> needed_kernels);
    ~AlgoApproximate() = default;
    int upload(std::shared_ptr<Tensor> y, std::shared_ptr<Tensor> alpha, std::shared_ptr<Tensor> A, std::shared_ptr<Tensor> x,
               std::shared_ptr<Tensor> z, const CmdOpt& option) override;
    int run() override;
    int download(std::shared_ptr<Tensor>& result) const override;
    std::unordered_map<KernelType, std::shared_ptr<KernelInterface>> kernels_hashmap;
    std::shared_ptr<Tensor> pre_result = nullptr;
    std::shared_ptr<Tensor> cur_result = nullptr;
    std::vector<bool> if_active;
    double _eps;
    int _inactive_tolerance;
    double _terminate_active_rate;
    int _terminate_min;
};