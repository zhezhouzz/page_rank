#pragma once

#include "tensor/tensor.h"
#include "kernels/kernel_interface.h"
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include "opencl_utils.h"
#include <vector>

class KernelOpencl final : public KernelInterface {
public:
    KernelOpencl();
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
    ~KernelOpencl();

    std::shared_ptr<Tensor> _x;
    std::shared_ptr<Tensor> _y;
    std::shared_ptr<Tensor> _A;
    std::shared_ptr<CppCLMem<double>> tensor_A_mem;
    std::vector<int> _history_active_table;
    std::shared_ptr<CppCLMem<int>> tensor_active_mem;

    cl_platform_id platform_id = nullptr;
    cl_device_id device_id;
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;
    cl_int ret_code = CL_SUCCESS;
    int tensor_x_length = 0;
    std::shared_ptr<CppCLMem<double>> tensor_x_mem;
    std::shared_ptr<CppCLMem<double>> tensor_y_mem;
    std::shared_ptr<CppCLMem<double>> tensor_z_mem;
    int A2_pos_length = 0;
    int A2_coord_length = 0;
    int A_vals_length = 0;
    std::shared_ptr<CppCLMem<int>> tensor_A2_pos_mem;
    std::shared_ptr<CppCLMem<int>> tensor_A2_coord_mem;
    std::shared_ptr<CppCLMem<double>> tensor_A_vals_mem;
    std::shared_ptr<CppCLContext> context;
    std::shared_ptr<CppCLCQ> command_queue;
    std::shared_ptr<CppCLProgram> program;

    std::shared_ptr<CppCLKernel> sparse_mxv_kernel;
    std::shared_ptr<CppCLKernel> dense_mxv_kernel;
    std::shared_ptr<CppCLKernel> approximate_mxv_kernel;
private:
    int gen_markov_matrix(std::shared_ptr<Tensor> alpha, std::shared_ptr<Tensor> A);
};