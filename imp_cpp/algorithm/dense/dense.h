#pragma once
#include <unordered_map>
#include "algorithm/algo_interface.h"

class AlgoDense final : public AlgoInterface {
public:
    AlgoDense(std::unordered_set<KernelType> needed_kernels);
    ~AlgoDense() = default;
    int upload(taco_tensor_t* y, taco_tensor_t* alpha, taco_tensor_t* A, taco_tensor_t* x,
               taco_tensor_t* z) override;
    int run() override;
    int download(taco_tensor_t** result) const override;
    std::unordered_map<KernelType, std::shared_ptr<KernelInterface>> kernels_hashmap;
    taco_tensor_t* pre_result = nullptr;
    taco_tensor_t* cur_result = nullptr;
};