#pragma once

#include <string>
#include <vector>
#include "debug/utils_debug.h"

enum class TENSOR_MODE { TENSOR_MODE_DENSE, TENSOR_MODE_SPARSE };

class Tensor {
public:
    Tensor(TENSOR_MODE mode_, const std::string &mtx_path);
    Tensor(TENSOR_MODE mode_, const std::vector<int> &dimensions_);
    Tensor(double v);
    void print(FpDebugLevel level = FP_LEVEL_WARNING);
    void save(const std::string &mtx_path);
    ~Tensor();
    std::vector<int> dimensions;
    std::vector<int> indices;
    int *cols = nullptr;
    uint8_t *vals;
    uint64_t vals_size;
    uint64_t unit_num;
    uint8_t unit_size;
    TENSOR_MODE mode;

private:
    int load_sparce_mtx(const std::string &mtx_path);
    int load_dense_mtx(const std::string &mtx_path);
    void print_(int d_index, int v_index, FpDebugLevel level);
    void save_(int d_index, int v_index, std::ofstream& ofstr);
};