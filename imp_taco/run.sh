#!/bin/bash

# clang++ -std=c++11 -DTACO page_rank.cpp -ltaco && ./a.out && cat y.tns

clang++ -std=c++14 -O3 -D CUR_DEBUG_LEVEL=$1 page_rank.cpp debug/utils_debug.cpp -ltaco && ./a.out && cat x.tns
