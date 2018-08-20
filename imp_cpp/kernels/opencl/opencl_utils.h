#pragma once

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include <fstream>
#include <iostream>
#include <string>
#include "cl_err_handle.h"
#include "debug/utils_debug.h"

#define CL_ERROR_HANDLE_                                                                          \
    do {                                                                                          \
        if (ret_code != 0) {                                                                      \
            std::cout << "[" << __FILE__ ":" << __LINE__ << "] error: " << ret_code << std::endl; \
            exit(0);                                                                              \
        }                                                                                         \
    } while (0)

int load_file(const char* file_path, char** source_str);

class CppCLContext {
public:
    CppCLContext(cl_device_id* device_id);
    ~CppCLContext();
    int ret_code = 0;
    cl_context context;
};

class CppCLCQ {
public:
    CppCLCQ(std::shared_ptr<CppCLContext> context, cl_device_id device_id);
    ~CppCLCQ();
    int ret_code = 0;
    std::shared_ptr<CppCLContext> context_;
    cl_command_queue command_queue;
};

class CppCLProgram {
public:
    CppCLProgram(std::shared_ptr<CppCLContext> context, cl_device_id* device_id,
                 const std::string& program_filepath);
    ~CppCLProgram();
    int ret_code = 0;
    std::shared_ptr<CppCLContext> context_;
    cl_program program;
};

class CppCLKernel {
public:
    CppCLKernel(std::shared_ptr<CppCLProgram> program, const std::string& kernel_name);
    ~CppCLKernel();
    int ret_code = 0;
    std::shared_ptr<CppCLProgram> program_;
    cl_kernel kernel;
};

template <class T>
class CppCLMem {
public:
    /* if data_pointer == nullptr, would not memcpy */
    CppCLMem(std::shared_ptr<CppCLContext> context, std::shared_ptr<CppCLCQ> command_queue,
             const T* data_pointer, int size, cl_mem_flags flag) {
        context_ = context;
        command_queue_ = command_queue;
        mem = clCreateBuffer(context->context, flag, size * sizeof(T), nullptr, &ret_code);
        CL_ERROR_HANDLE_;
        if (data_pointer != nullptr) {
            ret_code = clEnqueueWriteBuffer(command_queue->command_queue, mem, CL_TRUE, 0,
                                            size * sizeof(T), data_pointer, 0, nullptr, nullptr);
            CL_ERROR_HANDLE_;
        }
        return;
    }
    ~CppCLMem() {
        ret_code = clReleaseMemObject(mem);
        CL_ERROR_HANDLE_;
        return;
    }
    int ret_code = 0;
    std::shared_ptr<CppCLContext> context_;
    std::shared_ptr<CppCLCQ> command_queue_;
    cl_mem mem;
};