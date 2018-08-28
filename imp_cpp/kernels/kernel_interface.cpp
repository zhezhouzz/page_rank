#include "kernel_interface.h"
#include "debug/utils_debug.h"
#include "kernels/opencl/clhost.h"
#include "kernels/plain_cpu/cpu_page_rank.h"

std::shared_ptr<KernelInterface> KernelInterface::make(
    KernelType type) {
    if (type == KernelType::opencl) {
        return std::make_shared<KernelOpencl>();
    } else if (type == KernelType::cpu) {
        return std::make_shared<KernelCpu>();
    }
    return nullptr;
}