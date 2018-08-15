#pragma once

#include "kernels/kernel_interface.h"
#include "algorithm/algo_interface.h"
typedef struct {
    KernelType kernel_type;
    AlgoType algo_type;
} CmdOpt;

CmdOpt cmd_handle(int argc, char* argv[]);