#pragma once

#include <taco.h>
#include "kernels/kernel_interface.h"
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

class KernelOpencl final : public KernelInterface {
public:
    KernelOpencl();
    int upload(taco_tensor_t* y, taco_tensor_t* alpha, taco_tensor_t* A, taco_tensor_t* x,
               taco_tensor_t* z) override;
    int page_rank_once(bool flag_x2y) override;
    int upload_dense_mxv(taco_tensor_t* y, taco_tensor_t* alpha, taco_tensor_t* A, taco_tensor_t* x,
               taco_tensor_t* z) override;
    int dense_mxv(bool flag_x2y) override;
    int download(bool flag_x2y, taco_tensor_t** pre_result, taco_tensor_t** cur_result) override;
    double vetor_norm(taco_tensor_t* x, taco_tensor_t* y) const override;
    ~KernelOpencl();

    taco_tensor_t* x;
    taco_tensor_t* y;

    cl_platform_id platform_id = nullptr;
    cl_device_id device_id;
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;
    cl_int ret_code = CL_SUCCESS;
    int tensor_x_length = 0;
    cl_mem tensor_x_mem;
    cl_mem tensor_y_mem;
    cl_mem tensor_z_mem;
    int A2_pos_length = 0;
    int A2_coord_length = 0;
    int A_vals_length = 0;
    cl_mem tensor_A2_pos_mem;
    cl_mem tensor_A2_coord_mem;
    cl_mem tensor_A_vals_mem;
    cl_context context;
    cl_command_queue command_queue;
    cl_program program;
    cl_kernel sparse_mxv_kernel;
    cl_kernel dense_mxv_kernel;
private:
    int make_cl_kernel(cl_program* program, const std::string& cl_kernel_filepath);
};