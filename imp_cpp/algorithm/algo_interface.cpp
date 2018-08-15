#include "algo_interface.h"
#include "algorithm/sparse/sparse.h"
#include "algorithm/dense/dense.h"

std::shared_ptr<AlgoInterface> AlgoInterface::make(AlgoType type,
                                               std::unordered_set<KernelType> needed_kernels) {
    if (type == AlgoType::sparse) {
        return std::make_shared<AlgoSparse>(needed_kernels);
    } else if (type == AlgoType::dense) {
        return std::make_shared<AlgoDense>(needed_kernels);
    }
    return nullptr;
}