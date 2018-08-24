#!/bin/bash

mkdir -p build
cd build
cmake \
-DCUR_DEBUG_LEVEL:STRING=$1 \
-DCMAKE_BUILD_TYPE=$2 \
$3 \
..
make
make install DESTDIR="../install-dir"
cd -
