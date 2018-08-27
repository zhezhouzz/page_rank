#pragma once

#include <string>
#include <vector>

enum class TENSOR_MODE { TENSOR_MODE_DENSE, TENSOR_MODE_SPARSE };

class Tensor {
public:
    Tensor(TENSOR_MODE mode_, const std::string &mtx_path);
    Tensor(TENSOR_MODE mode_, const std::vector<uint32_t> &dimensions_);
    Tensor(double v);
    ~Tensor();
    std::vector<uint32_t> dimensions;
    std::vector<uint32_t> indices;
    uint8_t *vals;
    uint64_t vals_size;
    uint64_t unit_num;
    uint8_t unit_size;
    TENSOR_MODE mode;

private:
    int load_sparce_mtx(const std::string &mtx_path);
    int load_dense_mtx(const std::string &mtx_path);
};