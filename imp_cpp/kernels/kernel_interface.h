#pragma once

#include "tensor/tensor.h"
enum class KernelType { opencl, cpu };

class KernelInterface {
public:
    virtual ~KernelInterface() = default;
    virtual int upload(std::shared_ptr<Tensor> y, std::shared_ptr<Tensor> alpha,
                       std::shared_ptr<Tensor> A, std::shared_ptr<Tensor> x,
                       std::shared_ptr<Tensor> z) = 0;
    virtual int page_rank_once(bool flag_x2y) = 0;
    virtual int upload_dense_mxv(std::shared_ptr<Tensor> y, std::shared_ptr<Tensor> alpha,
                                 std::shared_ptr<Tensor> A, std::shared_ptr<Tensor> x,
                                 std::shared_ptr<Tensor> z) = 0;
    virtual int dense_mxv(bool flag_x2y) = 0;
    virtual int upload_approximate_mxv(std::shared_ptr<Tensor> y, std::shared_ptr<Tensor> alpha,
                                       std::shared_ptr<Tensor> A, std::shared_ptr<Tensor> x,
                                       std::shared_ptr<Tensor> z) = 0;
    virtual int approximate_mxv(bool flag_x2y, std::vector<bool> if_active) = 0;
    virtual int approximate_find_active(std::shared_ptr<Tensor> x, std::shared_ptr<Tensor> y,
                                        std::vector<bool>& if_active, double eps,
                                        int stable_num) = 0;
    virtual int normalize(bool flag_x2y, std::vector<bool>& if_active) = 0;

    virtual int download(bool flag_x2y, std::shared_ptr<Tensor>& pre_result,
                         std::shared_ptr<Tensor>& cur_result) = 0;
    virtual double vetor_norm(std::shared_ptr<Tensor> x, std::shared_ptr<Tensor> y) const = 0;
    static std::shared_ptr<KernelInterface> make(KernelType type);
};
