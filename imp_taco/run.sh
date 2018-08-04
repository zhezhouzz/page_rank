#!/bin/bash

clang++ -std=c++11 -DTACO page_rank.cpp -ltaco && ./a.out && cat y.tns
