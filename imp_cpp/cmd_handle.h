#pragma once

#include "kernels/kernel_wrapper.h"
typedef struct {
    KernelType kernel_type;
} CmdOpt;

CmdOpt cmd_handle(int argc, char* argv[]);