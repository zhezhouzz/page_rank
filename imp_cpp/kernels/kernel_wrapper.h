#pragma once

#include <taco.h>
enum class KernelType { opencl, taco };
int prepare(KernelType kernel_type);

int upload(taco_tensor_t* y, taco_tensor_t* alpha, taco_tensor_t* A, taco_tensor_t* x,
           taco_tensor_t* z, KernelType kernel_type);

int page_rank_once(taco_tensor_t* y, taco_tensor_t* alpha, taco_tensor_t* A, taco_tensor_t* x,
                   taco_tensor_t* z, KernelType kernel_type);

int download(taco_tensor_t** pre_result, taco_tensor_t** cur_result, KernelType kernel_type);

double vetor_norm(taco_tensor_t* x, taco_tensor_t* y, KernelType kernel_type);

int finish(KernelType kernel_type);