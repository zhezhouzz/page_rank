#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
#include <fstream>

#include <fstream>
#include "clhost.h"
#include "debug/utils_debug.h"
using namespace std;

constexpr int CL_WORK_GROUP = 1;

#define CL_ERR_HANDLE                                                                 \
    do {                                                                              \
        if (ret != CL_SUCCESS) {                                                      \
            FP_LOG(FP_LEVEL_ERROR, "[%s:%d] cl err = %d\n", __FILE__, __LINE__, ret); \
        }                                                                             \
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

int prepare(taco_tensor_t* c_tensor_A, taco_tensor_t* c_tensor_alpha, taco_tensor_t* c_tensor_x,
            taco_tensor_t* c_tensor_y, taco_tensor_t* c_tensor_z) {
    char* source_str = nullptr;
    size_t source_size = load_file(
        "/Users/admin/workspace/page_rank/imp_cpp/kernels/opencl/page_rank_kernel.cl", &source_str);
    FP_LOG(FP_LEVEL_INFO, "source_size = %d\n", int(source_size));
    cl_platform_id platform_id = nullptr;
    cl_device_id device_id;
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;
    cl_int ret = CL_SUCCESS;
    ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    CL_ERR_HANDLE;
    ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ALL, 1, &device_id, &ret_num_devices);
    CL_ERR_HANDLE;
    cl_context context = clCreateContext(nullptr, 1, &device_id, nullptr, nullptr, &ret);
    CL_ERR_HANDLE;
    cl_command_queue command_queue = clCreateCommandQueue(context, device_id, 0, &ret);
    CL_ERR_HANDLE;

    int tensor_x_length = (int)(c_tensor_x->dimensions[c_tensor_x->mode_ordering[0]]);
    cl_mem tensor_x_mem =
        clCreateBuffer(context, CL_MEM_READ_WRITE, tensor_x_length * sizeof(double), nullptr, &ret);
    CL_ERR_HANDLE;
    ret = clEnqueueWriteBuffer(command_queue, tensor_x_mem, CL_TRUE, 0,
                               tensor_x_length * sizeof(double), c_tensor_x->vals, 0, nullptr,
                               nullptr);
    CL_ERR_HANDLE;

    int tensor_z_length = (int)(c_tensor_z->dimensions[c_tensor_z->mode_ordering[0]]);
    cl_mem tensor_z_mem =
        clCreateBuffer(context, CL_MEM_READ_ONLY, tensor_z_length * sizeof(double), nullptr, &ret);
    CL_ERR_HANDLE;
    ret = clEnqueueWriteBuffer(command_queue, tensor_z_mem, CL_TRUE, 0,
                               tensor_z_length * sizeof(double), c_tensor_z->vals, 0, nullptr,
                               nullptr);
    CL_ERR_HANDLE;

    int* A1_pos = (int*)(c_tensor_A->indices[0][0]);
    int* A2_pos = (int*)(c_tensor_A->indices[1][0]);
    int A2_pos_length = A1_pos[1] + 1;
    int* A2_coord = (int*)(c_tensor_A->indices[1][1]);
    int A2_coord_length = A2_pos[A2_pos_length - 1];
    double* A_vals = (double*)(c_tensor_A->vals);
    int A_vals_length = A2_coord_length;
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

    cl_mem tensor_A2_pos_mem =
        clCreateBuffer(context, CL_MEM_READ_ONLY, A2_pos_length * sizeof(int), nullptr, &ret);
    CL_ERR_HANDLE;
    cl_mem tensor_A2_coord_mem =
        clCreateBuffer(context, CL_MEM_READ_ONLY, A2_coord_length * sizeof(int), nullptr, &ret);
    CL_ERR_HANDLE;
    cl_mem tensor_A_vals_mem =
        clCreateBuffer(context, CL_MEM_READ_ONLY, A_vals_length * sizeof(double), nullptr, &ret);
    CL_ERR_HANDLE;
    ret = clEnqueueWriteBuffer(command_queue, tensor_A2_pos_mem, CL_TRUE, 0,
                               A2_pos_length * sizeof(int), A2_pos, 0, nullptr, nullptr);
    CL_ERR_HANDLE;
    ret = clEnqueueWriteBuffer(command_queue, tensor_A2_coord_mem, CL_TRUE, 0,
                               A2_coord_length * sizeof(int), A2_coord, 0, nullptr, nullptr);
    CL_ERR_HANDLE;
    ret = clEnqueueWriteBuffer(command_queue, tensor_A_vals_mem, CL_TRUE, 0,
                               A_vals_length * sizeof(double), A_vals, 0, nullptr, nullptr);
    CL_ERR_HANDLE;

    int tensor_y_length = (int)(c_tensor_y->dimensions[c_tensor_y->mode_ordering[0]]);
    cl_mem tensor_y_mem =
        clCreateBuffer(context, CL_MEM_READ_WRITE, tensor_y_length * sizeof(double), nullptr, &ret);
    CL_ERR_HANDLE;

    cl_program program = clCreateProgramWithSource(context, 1, (const char**)&source_str,
                                                   (const size_t*)&source_size, &ret);
    CL_ERR_HANDLE;
    ret = clBuildProgram(program, 1, &device_id, nullptr, nullptr, nullptr);
    CL_ERR_HANDLE;
    cl_kernel kernel = clCreateKernel(program, "page_rank", &ret);
    CL_ERR_HANDLE;

    ret = clSetKernelArg(kernel, 0, sizeof(uint32_t), &tensor_x_length);
    CL_ERR_HANDLE;
    ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), &tensor_A2_pos_mem);
    CL_ERR_HANDLE;
    ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), &tensor_A2_coord_mem);
    CL_ERR_HANDLE;
    ret = clSetKernelArg(kernel, 3, sizeof(cl_mem), &tensor_A_vals_mem);
    CL_ERR_HANDLE;
    ret = clSetKernelArg(kernel, 4, sizeof(double) * CL_WORK_GROUP, nullptr);
    CL_ERR_HANDLE;
    ret = clSetKernelArg(kernel, 7, sizeof(cl_mem), &tensor_z_mem);
    CL_ERR_HANDLE;
    ret = clSetKernelArg(kernel, 8, sizeof(cl_mem), (double*)(c_tensor_alpha->vals));
    CL_ERR_HANDLE;
    size_t global_work_size[] = {static_cast<size_t>(CL_WORK_GROUP * tensor_x_length)};
    FP_LOG(FP_LEVEL_INFO, "global_work_size = %d\n", CL_WORK_GROUP);
    size_t local_work_size[] = {CL_WORK_GROUP};

    bool flag_x2y = true;
    for (int i = 0; i < 23; i++) {
        if (flag_x2y) {
            ret = clSetKernelArg(kernel, 5, sizeof(cl_mem), &tensor_x_mem);
            CL_ERR_HANDLE;
            ret = clSetKernelArg(kernel, 6, sizeof(cl_mem), &tensor_y_mem);
            CL_ERR_HANDLE;
        } else {
            ret = clSetKernelArg(kernel, 6, sizeof(cl_mem), &tensor_x_mem);
            CL_ERR_HANDLE;
            ret = clSetKernelArg(kernel, 5, sizeof(cl_mem), &tensor_y_mem);
            CL_ERR_HANDLE;
        }
        ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, nullptr, global_work_size,
                                     local_work_size, 0, nullptr, nullptr);
        CL_ERR_HANDLE;
        flag_x2y = not flag_x2y;
    }
    ret = clFlush(command_queue);
    CL_ERR_HANDLE;
    ret = clFinish(command_queue);
    CL_ERR_HANDLE;
    if (flag_x2y) {
        clEnqueueReadBuffer(command_queue, tensor_y_mem, CL_TRUE, 0,
                            tensor_x_length * sizeof(double), c_tensor_x->vals, 0, nullptr,
                            nullptr);
    } else {
        clEnqueueReadBuffer(command_queue, tensor_y_mem, CL_TRUE, 0,
                            tensor_x_length * sizeof(double), c_tensor_y->vals, 0, nullptr,
                            nullptr);
    }
    CL_ERR_HANDLE;
    return 0;
}

int run() { return 0; }

int finish() { return 0; }