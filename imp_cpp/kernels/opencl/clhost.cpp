#include "clhost.h"
#include <fstream>
#include "debug/utils_debug.h"

using namespace std;

constexpr int CL_WORK_GROUP = 1;

#define CL_ERR_HANDLE                                                                      \
    do {                                                                                   \
        if (ret_code != CL_SUCCESS) {                                                      \
            FP_LOG(FP_LEVEL_ERROR, "[%s:%d] cl err = %d\n", __FILE__, __LINE__, ret_code); \
        }                                                                                  \
    } while (0)

// static int load_file(const char* file_path, char** source_str) {
//     streampos size = 0;

//     ifstream file(file_path, ios::in | ios::binary | ios::ate);
//     if (file.is_open()) {
//         size = file.tellg();
//         *source_str = new char[size];
//         file.seekg(0, ios::beg);
//         file.read(*source_str, size);
//         file.close();
//     } else {
//         FP_LOG(FP_LEVEL_ERROR, "[%s:%d] load_file(%s) err\n", __FILE__, __LINE__, file_path);
//     }
//     return size;
// }

KernelOpencl::KernelOpencl() {
    ret_code = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    CL_ERR_HANDLE;
    ret_code = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ALL, 1, &device_id, &ret_num_devices);
    CL_ERR_HANDLE;
    context = std::make_shared<CppCLContext>(&device_id);
    command_queue = std::make_shared<CppCLCQ>(context, device_id);
    program = std::make_shared<CppCLProgram>(
        context, &device_id,
        "/Users/admin/workspace/page_rank/imp_cpp/kernels/opencl/page_rank_kernel.cl");
    sparse_mxv_kernel = std::make_shared<CppCLKernel>(program, "sparse_mxv");
    dense_mxv_kernel = std::make_shared<CppCLKernel>(program, "dense_mxv");
    return;
}

int KernelOpencl::upload(taco_tensor_t* c_tensor_y, taco_tensor_t* c_tensor_alpha,
                         taco_tensor_t* c_tensor_A, taco_tensor_t* c_tensor_x,
                         taco_tensor_t* c_tensor_z) {
    x = c_tensor_x;
    y = c_tensor_y;
    tensor_x_length = (int)(c_tensor_x->dimensions[c_tensor_x->mode_ordering[0]]);
    tensor_x_mem = std::make_shared<CppCLMem<double>>(
        context, command_queue, (double*)(c_tensor_x->vals), tensor_x_length, CL_MEM_READ_WRITE);

    int tensor_z_length = (int)(c_tensor_z->dimensions[c_tensor_z->mode_ordering[0]]);
    tensor_z_mem = std::make_shared<CppCLMem<double>>(
        context, command_queue, (double*)(c_tensor_z->vals), tensor_z_length, CL_MEM_READ_ONLY);

    int* A1_pos = (int*)(c_tensor_A->indices[0][0]);
    int* A2_pos = (int*)(c_tensor_A->indices[1][0]);
    A2_pos_length = A1_pos[1] + 1;
    int* A2_coord = (int*)(c_tensor_A->indices[1][1]);
    A2_coord_length = A2_pos[A2_pos_length - 1];
    double* A_vals = (double*)(c_tensor_A->vals);
    A_vals_length = A2_coord_length;
    FP_LOG(FP_LEVEL_INFO, "A2_pos_length = %d\n", A2_pos_length);
    FP_LOG(FP_LEVEL_INFO, "A2_pos: [ ");
    for (int i = 0; i < A2_pos_length; i++) {
        FP_LOG(FP_LEVEL_INFO, "%d ", A2_pos[i]);
    }
    FP_LOG(FP_LEVEL_INFO, "]\n");
    FP_LOG(FP_LEVEL_INFO, "A2_coord_length = %d\n", A2_coord_length);
    FP_LOG(FP_LEVEL_INFO, "A2_coord: [ ");
    for (int i = 0; i < A2_coord_length; i++) {
        FP_LOG(FP_LEVEL_INFO, "%d ", A2_coord[i]);
    }
    FP_LOG(FP_LEVEL_INFO, "]\n");

    tensor_A2_pos_mem = std::make_shared<CppCLMem<int>>(context, command_queue, A2_pos,
                                                        A2_pos_length, CL_MEM_READ_ONLY);
    tensor_A2_coord_mem = std::make_shared<CppCLMem<int>>(context, command_queue, A2_coord,
                                                          A2_coord_length, CL_MEM_READ_ONLY);
    tensor_A_vals_mem = std::make_shared<CppCLMem<double>>(context, command_queue, A_vals,
                                                           A_vals_length, CL_MEM_READ_ONLY);

    int tensor_y_length = (int)(c_tensor_y->dimensions[c_tensor_y->mode_ordering[0]]);
    tensor_y_mem = std::make_shared<CppCLMem<double>>(context, command_queue, nullptr,
                                                      tensor_y_length, CL_MEM_READ_WRITE);

    CL_ERR_HANDLE;
    ret_code = clSetKernelArg(sparse_mxv_kernel->kernel, 0, sizeof(uint32_t), &tensor_x_length);
    CL_ERR_HANDLE;
    ret_code =
        clSetKernelArg(sparse_mxv_kernel->kernel, 1, sizeof(cl_mem), &tensor_A2_pos_mem->mem);
    CL_ERR_HANDLE;
    ret_code =
        clSetKernelArg(sparse_mxv_kernel->kernel, 2, sizeof(cl_mem), &tensor_A2_coord_mem->mem);
    CL_ERR_HANDLE;
    ret_code =
        clSetKernelArg(sparse_mxv_kernel->kernel, 3, sizeof(cl_mem), &tensor_A_vals_mem->mem);
    CL_ERR_HANDLE;
    ret_code =
        clSetKernelArg(sparse_mxv_kernel->kernel, 4, sizeof(double) * CL_WORK_GROUP, nullptr);
    CL_ERR_HANDLE;
    ret_code = clSetKernelArg(sparse_mxv_kernel->kernel, 7, sizeof(cl_mem), &tensor_z_mem->mem);
    CL_ERR_HANDLE;
    ret_code = clSetKernelArg(sparse_mxv_kernel->kernel, 8, sizeof(cl_mem),
                              (double*)(c_tensor_alpha->vals));
    CL_ERR_HANDLE;
    return 0;
}

int KernelOpencl::page_rank_once(bool flag_x2y) {
    size_t global_work_size[] = {static_cast<size_t>(CL_WORK_GROUP * tensor_x_length)};
    FP_LOG(FP_LEVEL_INFO, "global_work_size = %d\n", CL_WORK_GROUP);
    size_t local_work_size[] = {CL_WORK_GROUP};
    if (flag_x2y) {
        ret_code = clSetKernelArg(sparse_mxv_kernel->kernel, 5, sizeof(cl_mem), &tensor_x_mem->mem);
        CL_ERR_HANDLE;
        ret_code = clSetKernelArg(sparse_mxv_kernel->kernel, 6, sizeof(cl_mem), &tensor_y_mem->mem);
        CL_ERR_HANDLE;
    } else {
        ret_code = clSetKernelArg(sparse_mxv_kernel->kernel, 6, sizeof(cl_mem), &tensor_x_mem->mem);
        CL_ERR_HANDLE;
        ret_code = clSetKernelArg(sparse_mxv_kernel->kernel, 5, sizeof(cl_mem), &tensor_y_mem->mem);
        CL_ERR_HANDLE;
    }
    ret_code =
        clEnqueueNDRangeKernel(command_queue->command_queue, sparse_mxv_kernel->kernel, 1, nullptr,
                               global_work_size, local_work_size, 0, nullptr, nullptr);
    CL_ERR_HANDLE;
    return 0;
}

int KernelOpencl::upload_dense_mxv(taco_tensor_t* c_tensor_y, taco_tensor_t* c_tensor_alpha,
                                   taco_tensor_t* c_tensor_A, taco_tensor_t* c_tensor_x,
                                   taco_tensor_t* c_tensor_z) {
    x = c_tensor_x;
    y = c_tensor_y;
    tensor_x_length = (int)(c_tensor_x->dimensions[c_tensor_x->mode_ordering[0]]);
    tensor_x_mem = std::make_shared<CppCLMem<double>>(
        context, command_queue, (double*)(c_tensor_x->vals), tensor_x_length, CL_MEM_READ_WRITE);

    int tensor_z_length = (int)(c_tensor_z->dimensions[c_tensor_z->mode_ordering[0]]);
    tensor_z_mem = std::make_shared<CppCLMem<double>>(
        context, command_queue, (double*)(c_tensor_z->vals), tensor_z_length, CL_MEM_READ_ONLY);

    int A1_dimension = (int)(c_tensor_A->dimensions[c_tensor_A->mode_ordering[0]]);
    int A2_dimension = (int)(c_tensor_A->dimensions[c_tensor_A->mode_ordering[1]]);
    double* A_vals = (double*)(c_tensor_A->vals);
    tensor_A_vals_mem = std::make_shared<CppCLMem<double>>(
        context, command_queue, A_vals, A1_dimension * A2_dimension, CL_MEM_READ_ONLY);

    int tensor_y_length = (int)(c_tensor_y->dimensions[c_tensor_y->mode_ordering[0]]);
    tensor_y_mem = std::make_shared<CppCLMem<double>>(context, command_queue, nullptr,
                                                      tensor_y_length, CL_MEM_READ_WRITE);

    ret_code = clSetKernelArg(dense_mxv_kernel->kernel, 0, sizeof(uint32_t), &tensor_x_length);
    CL_ERR_HANDLE;
    ret_code = clSetKernelArg(dense_mxv_kernel->kernel, 1, sizeof(cl_mem), &tensor_A_vals_mem->mem);
    CL_ERR_HANDLE;
    ret_code = clSetKernelArg(dense_mxv_kernel->kernel, 2, sizeof(double) * CL_WORK_GROUP, nullptr);
    CL_ERR_HANDLE;
    ret_code = clSetKernelArg(dense_mxv_kernel->kernel, 5, sizeof(cl_mem), &tensor_z_mem->mem);
    CL_ERR_HANDLE;
    ret_code = clSetKernelArg(dense_mxv_kernel->kernel, 6, sizeof(cl_mem),
                              (double*)(c_tensor_alpha->vals));
    CL_ERR_HANDLE;
    return 0;
}
int KernelOpencl::dense_mxv(bool flag_x2y) {
    size_t global_work_size[] = {static_cast<size_t>(CL_WORK_GROUP * tensor_x_length)};
    FP_LOG(FP_LEVEL_INFO, "global_work_size = %d\n", CL_WORK_GROUP);
    size_t local_work_size[] = {CL_WORK_GROUP};
    if (flag_x2y) {
        ret_code = clSetKernelArg(dense_mxv_kernel->kernel, 3, sizeof(cl_mem), &tensor_x_mem->mem);
        CL_ERR_HANDLE;
        ret_code = clSetKernelArg(dense_mxv_kernel->kernel, 4, sizeof(cl_mem), &tensor_y_mem->mem);
        CL_ERR_HANDLE;
    } else {
        ret_code = clSetKernelArg(dense_mxv_kernel->kernel, 4, sizeof(cl_mem), &tensor_x_mem->mem);
        CL_ERR_HANDLE;
        ret_code = clSetKernelArg(dense_mxv_kernel->kernel, 3, sizeof(cl_mem), &tensor_y_mem->mem);
        CL_ERR_HANDLE;
    }
    ret_code =
        clEnqueueNDRangeKernel(command_queue->command_queue, dense_mxv_kernel->kernel, 1, nullptr,
                               global_work_size, local_work_size, 0, nullptr, nullptr);
    CL_ERR_HANDLE;
    return 0;
}

int KernelOpencl::download(bool flag_x2y, taco_tensor_t** c_tensor_x, taco_tensor_t** c_tensor_y) {
    ret_code = clFlush(command_queue->command_queue);
    CL_ERR_HANDLE;
    ret_code = clFinish(command_queue->command_queue);
    CL_ERR_HANDLE;
    if (flag_x2y) {
        clEnqueueReadBuffer(command_queue->command_queue, tensor_x_mem->mem, CL_TRUE, 0,
                            tensor_x_length * sizeof(double), y->vals, 0, nullptr, nullptr);
        CL_ERR_HANDLE;
        clEnqueueReadBuffer(command_queue->command_queue, tensor_y_mem->mem, CL_TRUE, 0,
                            tensor_x_length * sizeof(double), x->vals, 0, nullptr, nullptr);
        CL_ERR_HANDLE;
    } else {
        clEnqueueReadBuffer(command_queue->command_queue, tensor_y_mem->mem, CL_TRUE, 0,
                            tensor_x_length * sizeof(double), y->vals, 0, nullptr, nullptr);
        CL_ERR_HANDLE;
        clEnqueueReadBuffer(command_queue->command_queue, tensor_x_mem->mem, CL_TRUE, 0,
                            tensor_x_length * sizeof(double), x->vals, 0, nullptr, nullptr);
        CL_ERR_HANDLE;
    }
    *c_tensor_x = x;
    *c_tensor_y = y;
    return 0;
}

double KernelOpencl::vetor_norm(taco_tensor_t* x, taco_tensor_t* y) const {
    int y1_dimension = (int)(y->dimensions[y->mode_ordering[0]]);
    int x1_dimension = (int)(x->dimensions[x->mode_ordering[0]]);
    assert(y1_dimension == x1_dimension);
    double* __restrict x_vals = (double*)(x->vals);
    double* __restrict y_vals = (double*)(y->vals);
    double norm = 0;
#pragma omp parallel for
    for (int32_t iy = 0; iy < y1_dimension; iy++) {
        double diff = x_vals[iy] - y_vals[iy];
        norm += diff * diff;
    }
    return norm;
}

KernelOpencl::~KernelOpencl() { return; }