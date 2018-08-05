#!/bin/bash

# clang++ -std=c++11 -DTACO page_rank.cpp -ltaco && ./a.out && cat y.tns

clang++ -std=c++14 -O3 page_rank.cpp  -ltaco && ./a.out && cat x.tns
