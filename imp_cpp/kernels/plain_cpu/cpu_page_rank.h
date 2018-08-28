#include "tensor/tensor.h"
#include "kernels/kernel_interface.h"

class KernelCpu final : public KernelInterface {
public:
    KernelCpu() = default;
    int upload(std::shared_ptr<Tensor> y, std::shared_ptr<Tensor> alpha, std::shared_ptr<Tensor> A, std::shared_ptr<Tensor> x,
               std::shared_ptr<Tensor> z) override;
    int page_rank_once(bool flag_x2y) override;
    int upload_dense_mxv(std::shared_ptr<Tensor> y, std::shared_ptr<Tensor> alpha, std::shared_ptr<Tensor> A, std::shared_ptr<Tensor> x,
                         std::shared_ptr<Tensor> z) override;
    int dense_mxv(bool flag_x2y) override;
    int upload_approximate_mxv(std::shared_ptr<Tensor> y, std::shared_ptr<Tensor> alpha, std::shared_ptr<Tensor> A,
                               std::shared_ptr<Tensor> x, std::shared_ptr<Tensor> z) override;
    int approximate_mxv(bool flag_x2y, std::vector<bool> if_active) override;
    int approximate_find_active(std::shared_ptr<Tensor> x, std::shared_ptr<Tensor> y, std::vector<bool>& if_active,
                                double eps, int stable_num) override;
    int normalize(bool flag_x2y, std::vector<bool>& if_active) override;

    int download(bool flag_x2y, std::shared_ptr<Tensor>& pre_result, std::shared_ptr<Tensor>& cur_result) override;
    double vetor_norm(std::shared_ptr<Tensor> x, std::shared_ptr<Tensor> y) const override;
    ~KernelCpu() = default;

    std::shared_ptr<Tensor> _y;
    std::shared_ptr<Tensor> _alpha;
    std::shared_ptr<Tensor> _A;
    std::shared_ptr<Tensor> _x;
    std::shared_ptr<Tensor> _z;
    std::vector<int> _history_active_table;
};

int swap_vector(std::shared_ptr<Tensor> x, std::shared_ptr<Tensor> y);