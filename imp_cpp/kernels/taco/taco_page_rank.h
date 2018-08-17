#include <taco.h>
#include "kernels/kernel_interface.h"

class KernelTaco final : public KernelInterface {
public:
    KernelTaco() = default;
    int upload(taco_tensor_t* y, taco_tensor_t* alpha, taco_tensor_t* A, taco_tensor_t* x,
               taco_tensor_t* z) override;
    int page_rank_once(bool flag_x2y) override;
    int upload_dense_mxv(taco_tensor_t* y, taco_tensor_t* alpha, taco_tensor_t* A, taco_tensor_t* x,
                         taco_tensor_t* z) override;
    int dense_mxv(bool flag_x2y) override;
    int upload_approximate_mxv(taco_tensor_t* y, taco_tensor_t* alpha, taco_tensor_t* A,
                               taco_tensor_t* x, taco_tensor_t* z) override;
    int approximate_mxv(bool flag_x2y, std::vector<bool> if_active) override;
    int approximate_find_active(taco_tensor_t* x, taco_tensor_t* y, std::vector<bool>& if_active,
                                double eps, int stable_num) override;
    int normalize(bool flag_x2y, std::vector<bool>& if_active) override;

    int download(bool flag_x2y, taco_tensor_t** pre_result, taco_tensor_t** cur_result) override;
    double vetor_norm(taco_tensor_t* x, taco_tensor_t* y) const override;
    ~KernelTaco() = default;

    taco_tensor_t* _y;
    taco_tensor_t* _alpha;
    taco_tensor_t* _A;
    taco_tensor_t* _x;
    taco_tensor_t* _z;
    std::vector<int> _history_active_table;
};

int taco_swap_vector(taco_tensor_t* x, taco_tensor_t* y);