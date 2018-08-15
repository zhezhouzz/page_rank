#include "opencl_utils.h"

int load_file(const char* file_path, char** source_str) {
    std::streampos size = 0;

    std::ifstream file(file_path, std::ios::in | std::ios::binary | std::ios::ate);
    if (file.is_open()) {
        size = file.tellg();
        *source_str = new char[size];
        file.seekg(0, std::ios::beg);
        file.read(*source_str, size);
        file.close();
    } else {
        FP_LOG(FP_LEVEL_ERROR, "[%s:%d] load_file(%s) err\n", __FILE__, __LINE__, file_path);
    }
    return size;
}

CppCLContext::CppCLContext(cl_device_id* device_id) {
    context = clCreateContext(nullptr, 1, device_id, nullptr, nullptr, &ret_code);
    CL_ERROR_HANDLE_;
    return;
}

CppCLContext::~CppCLContext() {
    ret_code = clReleaseContext(context);
    CL_ERROR_HANDLE_;
    return;
}

CppCLCQ::CppCLCQ(std::shared_ptr<CppCLContext> context, cl_device_id device_id) {
    context_ = context;
    command_queue = clCreateCommandQueue(context->context, device_id, 0, &ret_code);
    CL_ERROR_HANDLE_;
    return;
}

CppCLCQ::~CppCLCQ() {
    ret_code = clReleaseCommandQueue(command_queue);
    CL_ERROR_HANDLE_;
    return;
}

CppCLProgram::CppCLProgram(std::shared_ptr<CppCLContext> context, cl_device_id* device_id,
                           const std::string& program_filepath) {
    context_ = context;
    char* source_str = nullptr;
    size_t source_size = load_file(program_filepath.c_str(), &source_str);
    FP_LOG(FP_LEVEL_INFO, "source_size = %d\n", int(source_size));
    program = clCreateProgramWithSource(context->context, 1, (const char**)&source_str,
                                        (const size_t*)&source_size, &ret_code);
    CL_ERROR_HANDLE_;
    ret_code = clBuildProgram(program, 1, device_id, nullptr, nullptr, nullptr);
    CL_ERROR_HANDLE_;
    return;
}

CppCLProgram::~CppCLProgram() {
    ret_code = clReleaseProgram(program);
    CL_ERROR_HANDLE_;
    return;
}

CppCLKernel::CppCLKernel(std::shared_ptr<CppCLProgram> program, const std::string& kernel_name) {
    program_ = program;
    kernel = clCreateKernel(program->program, kernel_name.c_str(), &ret_code);
    CL_ERROR_HANDLE_;
    return;
}

CppCLKernel::~CppCLKernel() {
    ret_code = clReleaseKernel(kernel);
    CL_ERROR_HANDLE_;
    return;
}
