#pragma once

constexpr double PAGE_RANK_D = 0.85f;
constexpr double PAGE_RANK_MAX = 1.0f;
// const char* const MTX_DATA_PATH const = "/Users/admin/workspace/page_rank/data/page_map.mtx";
const char* const MTX_DATA_PATH  = "/Users/admin/workspace/page_rank/data/page_map_dense.mtx";
// const char* const MTX_DATA_PATH  const= "/Users/admin/workspace/page_rank/data/8x8-12.mtx";
const char* const DEFAULT_KERNEL  = "cpu";
const char* const DEFAULT_ALGORITHM  = "sparse";
const char* const OPENCL_KERNEL_PATH  =
    "/Users/zhezhou/workspace/page_rank/imp_cpp/kernels/opencl/page_rank_kernel.cl";