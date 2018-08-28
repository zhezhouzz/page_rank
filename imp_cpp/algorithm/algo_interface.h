#pragma once

#include <unordered_set>
#include "kernels/kernel_interface.h"
#include "cmd/cmd_handle.h"
#include "algo_type.h"

constexpr double PAGE_RANK_EPS = 1.0e-8;

class AlgoInterface {
public:
    virtual ~AlgoInterface() = default;
    virtual int upload(std::shared_ptr<Tensor> y, std::shared_ptr<Tensor> alpha, std::shared_ptr<Tensor> A, std::shared_ptr<Tensor> x,
                       std::shared_ptr<Tensor> z, const CmdOpt& option) = 0;
    virtual int run() = 0;
    virtual int download(std::shared_ptr<Tensor>& result) const = 0;
    static std::shared_ptr<AlgoInterface> make(AlgoType type,
                                               std::unordered_set<KernelType> needed_kernels);
};