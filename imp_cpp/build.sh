#!/bin/bash

mkdir -p build
cd build
cmake \
-DCUR_DEBUG_LEVEL:STRING=2 \
-DUSE_OPENCL=ON \
-DCMAKE_BUILD_TYPE=Release \
..
make
make install DESTDIR="../install-dir"
cd -
