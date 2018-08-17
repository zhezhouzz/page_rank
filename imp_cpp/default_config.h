#pragma once

constexpr double PAGE_RANK_D = 0.85f;
constexpr double PAGE_RANK_MAX = 1.0f;
// const char* MTX_DATA_PATH = "/Users/admin/workspace/page_rank/data/page_map.mtx";
const char* MTX_DATA_PATH = "/Users/admin/workspace/page_rank/data/page_map_dense.mtx";
// const char* MTX_DATA_PATH = "/Users/admin/workspace/page_rank/data/8x8-12.mtx";
const char* DEFAULT_KERNEL = "taco";
const char* DEFAULT_ALGORITHM = "sparse";