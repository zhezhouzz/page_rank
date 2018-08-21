#pragma once

#include "kernels/kernel_interface.h"
#include "algorithm/algo_type.h"
#include <string>

typedef struct {
    KernelType kernel_type;
    AlgoType algo_type;
    std::string data_set_path;
    double eps;
    int inactive_tolerance;
    double terminate_active_rate;
} CmdOpt;

CmdOpt cmd_handle(int argc, char* argv[]);