#pragma once

#include <unordered_set>
#include "kernels/kernel_interface.h"

enum class AlgoType { sparse, dense };

constexpr double PAGE_RANK_EPS = 1.0e-8;

class AlgoInterface {
public:
    virtual ~AlgoInterface() = default;
    virtual int upload(taco_tensor_t* y, taco_tensor_t* alpha, taco_tensor_t* A, taco_tensor_t* x,
                       taco_tensor_t* z) = 0;
    virtual int run() = 0;
    virtual int download(taco_tensor_t** result) const = 0;
    static std::shared_ptr<AlgoInterface> make(AlgoType type,
                                               std::unordered_set<KernelType> needed_kernels);
};