#include "kernel_interface.h"
#include "debug/utils_debug.h"
#include "kernels/opencl/clhost.h"
#include "kernels/taco/taco_page_rank.h"

std::shared_ptr<KernelInterface> KernelInterface::make(
    KernelType type) {
    if (type == KernelType::opencl) {
        return std::make_shared<KernelOpencl>();
    } else if (type == KernelType::taco) {
        return std::make_shared<KernelTaco>();
    }
    return nullptr;
}