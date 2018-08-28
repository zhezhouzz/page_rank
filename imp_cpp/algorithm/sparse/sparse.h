#pragma once
#include <unordered_map>
#include "algorithm/algo_interface.h"

class AlgoSparse final : public AlgoInterface {
public:
    AlgoSparse(std::unordered_set<KernelType> needed_kernels);
    ~AlgoSparse() = default;
    int upload(std::shared_ptr<Tensor> y, std::shared_ptr<Tensor> alpha, std::shared_ptr<Tensor> A, std::shared_ptr<Tensor> x,
               std::shared_ptr<Tensor> z, const CmdOpt& option) override;
    int run() override;
    int download(std::shared_ptr<Tensor>& result) const override;
    std::unordered_map<KernelType, std::shared_ptr<KernelInterface>> kernels_hashmap;
    std::shared_ptr<Tensor> pre_result = nullptr;
    std::shared_ptr<Tensor> cur_result = nullptr;
    double _eps;
};