#include "kernel_wrapper.h"
#include "debug/utils_debug.h"
#include "kernels/opencl/clhost.h"
#include "kernels/taco/taco_page_rank.h"

constexpr int run_batch = 34;
static bool flag_x2y = true;

int prepare(KernelType kernel_type) {
    switch (kernel_type) {
        case KernelType::taco:
            return 0;
        case KernelType::opencl:
            return prepare();
    }
    return -1;
}

int upload(taco_tensor_t* y, taco_tensor_t* alpha, taco_tensor_t* A, taco_tensor_t* x,
           taco_tensor_t* z, KernelType kernel_type) {
    flag_x2y = true;
    switch (kernel_type) {
        case KernelType::taco:
            return 0;
        case KernelType::opencl:
            return upload(A, alpha, x, y, z);
    }
    return -1;
}

int page_rank_once(taco_tensor_t* y, taco_tensor_t* alpha, taco_tensor_t* A, taco_tensor_t* x,
                   taco_tensor_t* z, KernelType kernel_type) {
    switch (kernel_type) {
        case KernelType::taco:
            for (int i = 0; i < run_batch; i++) {
                int ret_code = taco_page_rank(flag_x2y);
                flag_x2y = not flag_x2y;
            }
            return 0;
        case KernelType::opencl:
            for (int i = 0; i < run_batch; i++) {
                run(flag_x2y);
                flag_x2y = not flag_x2y;
            }
            return 0;
    }
    return -1;
}

int download(taco_tensor_t** pre_result, taco_tensor_t** cur_result, KernelType kernel_type) {
    switch (kernel_type) {
        case KernelType::taco:
            return taco_download(flag_x2y, pre_result, cur_result);
        case KernelType::opencl:
            return download(flag_x2y, pre_result, cur_result);
    }
    return -1;
}

double vetor_norm(taco_tensor_t* x, taco_tensor_t* y, KernelType kernel_type) {
    switch (kernel_type) {
        case KernelType::taco:
            return taco_vetor_norm(x, y);
        case KernelType::opencl:
            return taco_vetor_norm(x, y);
    }
    return -1;
}

int finish(KernelType kernel_type) {
    switch (kernel_type) {
        case KernelType::taco:
            return 0;
        case KernelType::opencl:
            return finish();
    }
    return -1;
}