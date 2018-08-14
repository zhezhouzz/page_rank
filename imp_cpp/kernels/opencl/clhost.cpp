#include <fstream>
#include "clhost.h"
#include "debug/utils_debug.h"
using namespace std;

constexpr int CL_WORK_GROUP = 1;

#define CL_ERR_HANDLE                                                                        \
    do {                                                                                     \
        if (ret_code != CL_SUCCESS) {                                                      \
            FP_LOG(FP_LEVEL_ERROR, "[%s:%d] cl err = %d\n", __FILE__, __LINE__, ret_code); \
        }                                                                                    \
    } while (0)

static int load_file(const char* file_path, char** source_str) {
    streampos size = 0;

    ifstream file(file_path, ios::in | ios::binary | ios::ate);
    if (file.is_open()) {
        size = file.tellg();
        *source_str = new char[size];
        file.seekg(0, ios::beg);
        file.read(*source_str, size);
        file.close();
    } else {
        FP_LOG(FP_LEVEL_ERROR, "[%s:%d] load_file(%s) err\n", __FILE__, __LINE__, file_path);
    }
    return size;
}

KernelOpencl::KernelOpencl() {
    char* source_str = nullptr;
    size_t source_size = load_file(
        "/Users/admin/workspace/page_rank/imp_cpp/kernels/opencl/page_rank_kernel.cl", &source_str);
    FP_LOG(FP_LEVEL_INFO, "source_size = %d\n", int(source_size));
    ret_code = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    CL_ERR_HANDLE;
    ret_code =
        clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ALL, 1, &device_id, &ret_num_devices);
    CL_ERR_HANDLE;
    context = clCreateContext(nullptr, 1, &device_id, nullptr, nullptr, &ret_code);
    CL_ERR_HANDLE;
    command_queue = clCreateCommandQueue(context, device_id, 0, &ret_code);
    CL_ERR_HANDLE;
    program = clCreateProgramWithSource(context, 1, (const char**)&source_str,
                                          (const size_t*)&source_size, &ret_code);
    CL_ERR_HANDLE;
    ret_code = clBuildProgram(program, 1, &device_id, nullptr, nullptr, nullptr);
    CL_ERR_HANDLE;
    kernel = clCreateKernel(program, "page_rank", &ret_code);
    CL_ERR_HANDLE;

    return;
}

int KernelOpencl::upload(taco_tensor_t* c_tensor_y, taco_tensor_t* c_tensor_alpha,
                         taco_tensor_t* c_tensor_A, taco_tensor_t* c_tensor_x,
                         taco_tensor_t* c_tensor_z) {
    x = c_tensor_x;
    y = c_tensor_y;
    tensor_x_length = (int)(c_tensor_x->dimensions[c_tensor_x->mode_ordering[0]]);
    tensor_x_mem = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                    tensor_x_length * sizeof(double), nullptr, &ret_code);
    CL_ERR_HANDLE;
    ret_code = clEnqueueWriteBuffer(command_queue, tensor_x_mem, CL_TRUE, 0,
                                      tensor_x_length * sizeof(double), c_tensor_x->vals, 0,
                                      nullptr, nullptr);
    CL_ERR_HANDLE;

    int tensor_z_length = (int)(c_tensor_z->dimensions[c_tensor_z->mode_ordering[0]]);
    tensor_z_mem = clCreateBuffer(context, CL_MEM_READ_ONLY, tensor_z_length * sizeof(double),
                                    nullptr, &ret_code);
    CL_ERR_HANDLE;
    ret_code = clEnqueueWriteBuffer(command_queue, tensor_z_mem, CL_TRUE, 0,
                                      tensor_z_length * sizeof(double), c_tensor_z->vals, 0,
                                      nullptr, nullptr);
    CL_ERR_HANDLE;

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

    tensor_A2_pos_mem = clCreateBuffer(context, CL_MEM_READ_ONLY, A2_pos_length * sizeof(int),
                                         nullptr, &ret_code);
    CL_ERR_HANDLE;
    tensor_A2_coord_mem = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                           A2_coord_length * sizeof(int), nullptr, &ret_code);
    CL_ERR_HANDLE;
    tensor_A_vals_mem = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                         A_vals_length * sizeof(double), nullptr, &ret_code);
    CL_ERR_HANDLE;
    ret_code = clEnqueueWriteBuffer(command_queue, tensor_A2_pos_mem, CL_TRUE, 0,
                                      A2_pos_length * sizeof(int), A2_pos, 0, nullptr, nullptr);
    CL_ERR_HANDLE;
    ret_code =
        clEnqueueWriteBuffer(command_queue, tensor_A2_coord_mem, CL_TRUE, 0,
                             A2_coord_length * sizeof(int), A2_coord, 0, nullptr, nullptr);
    CL_ERR_HANDLE;
    ret_code =
        clEnqueueWriteBuffer(command_queue, tensor_A_vals_mem, CL_TRUE, 0,
                             A_vals_length * sizeof(double), A_vals, 0, nullptr, nullptr);
    CL_ERR_HANDLE;

    int tensor_y_length = (int)(c_tensor_y->dimensions[c_tensor_y->mode_ordering[0]]);
    tensor_y_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, tensor_y_length * sizeof(double),
                                    nullptr, &ret_code);
    CL_ERR_HANDLE;
    ret_code = clSetKernelArg(kernel, 0, sizeof(uint32_t), &tensor_x_length);
    CL_ERR_HANDLE;
    ret_code = clSetKernelArg(kernel, 1, sizeof(cl_mem), &tensor_A2_pos_mem);
    CL_ERR_HANDLE;
    ret_code = clSetKernelArg(kernel, 2, sizeof(cl_mem), &tensor_A2_coord_mem);
    CL_ERR_HANDLE;
    ret_code = clSetKernelArg(kernel, 3, sizeof(cl_mem), &tensor_A_vals_mem);
    CL_ERR_HANDLE;
    ret_code = clSetKernelArg(kernel, 4, sizeof(double) * CL_WORK_GROUP, nullptr);
    CL_ERR_HANDLE;
    ret_code = clSetKernelArg(kernel, 7, sizeof(cl_mem), &tensor_z_mem);
    CL_ERR_HANDLE;
    ret_code = clSetKernelArg(kernel, 8, sizeof(cl_mem), (double*)(c_tensor_alpha->vals));
    CL_ERR_HANDLE;
    return 0;
}

int KernelOpencl::page_rank_once(bool flag_x2y) {
    size_t global_work_size[] = {static_cast<size_t>(CL_WORK_GROUP * tensor_x_length)};
    FP_LOG(FP_LEVEL_INFO, "global_work_size = %d\n", CL_WORK_GROUP);
    size_t local_work_size[] = {CL_WORK_GROUP};
    if (flag_x2y) {
        ret_code = clSetKernelArg(kernel, 5, sizeof(cl_mem), &tensor_x_mem);
        CL_ERR_HANDLE;
        ret_code = clSetKernelArg(kernel, 6, sizeof(cl_mem), &tensor_y_mem);
        CL_ERR_HANDLE;
    } else {
        ret_code = clSetKernelArg(kernel, 6, sizeof(cl_mem), &tensor_x_mem);
        CL_ERR_HANDLE;
        ret_code = clSetKernelArg(kernel, 5, sizeof(cl_mem), &tensor_y_mem);
        CL_ERR_HANDLE;
    }
    ret_code = clEnqueueNDRangeKernel(command_queue, kernel, 1, nullptr, global_work_size,
                                        local_work_size, 0, nullptr, nullptr);
    CL_ERR_HANDLE;
    return 0;
}

int KernelOpencl::download(bool flag_x2y, taco_tensor_t** c_tensor_x, taco_tensor_t** c_tensor_y) {
    ret_code = clFlush(command_queue);
    CL_ERR_HANDLE;
    ret_code = clFinish(command_queue);
    CL_ERR_HANDLE;
    if (flag_x2y) {
        clEnqueueReadBuffer(command_queue, tensor_x_mem, CL_TRUE, 0,
                            tensor_x_length * sizeof(double), y->vals, 0, nullptr, nullptr);
        CL_ERR_HANDLE;
        clEnqueueReadBuffer(command_queue, tensor_y_mem, CL_TRUE, 0,
                            tensor_x_length * sizeof(double), x->vals, 0, nullptr, nullptr);
        CL_ERR_HANDLE;
    } else {
        clEnqueueReadBuffer(command_queue, tensor_y_mem, CL_TRUE, 0,
                            tensor_x_length * sizeof(double), y->vals, 0, nullptr, nullptr);
        CL_ERR_HANDLE;
        clEnqueueReadBuffer(command_queue, tensor_x_mem, CL_TRUE, 0,
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

KernelOpencl::~KernelOpencl() {
    ret_code = clReleaseKernel(kernel);
    CL_ERR_HANDLE;
    ret_code = clReleaseProgram(program);
    CL_ERR_HANDLE;
    ret_code = clReleaseMemObject(tensor_A2_pos_mem);
    CL_ERR_HANDLE;
    ret_code = clReleaseMemObject(tensor_A2_coord_mem);
    CL_ERR_HANDLE;
    ret_code = clReleaseMemObject(tensor_A_vals_mem);
    CL_ERR_HANDLE;
    ret_code = clReleaseMemObject(tensor_x_mem);
    CL_ERR_HANDLE;
    ret_code = clReleaseMemObject(tensor_y_mem);
    CL_ERR_HANDLE;
    ret_code = clReleaseMemObject(tensor_z_mem);
    CL_ERR_HANDLE;
    ret_code = clReleaseCommandQueue(command_queue);
    CL_ERR_HANDLE;
    ret_code = clReleaseContext(context);
    CL_ERR_HANDLE;
    return;
}