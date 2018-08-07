#!/bin/bash

# clang++ -std=c++11 -DTACO page_rank.cpp -ltaco && ./a.out && cat y.tns

CPPFLAGS=""
CPPFLAGS+=" -std=c++14 "
CPPFLAGS+=" -O3 "
CPPFLAGS+=" -D CUR_DEBUG_LEVEL="$1" "
CPPFLAGS+=" -ltaco "
CPPFLAGS+=" -I. "

#if use openmp
if [ "$2" = "openmp" ]; then
    echo "enable openmp"
    CPPFLAGS+="-fopenmp -L/usr/local/opt/libomp/lib -I/usr/local/opt/libomp/include  -Xpreprocessor -fopenmp -lomp"
elif [ "$2" = "opencl" ] ; then
    echo "enable opencl"
    CPPFLAGS+=" -framework OpenCL "
    CPPFLAGS+=" -D USE_OPENCL=ON "
    CPPFLAGS+=" kernels/opencl/clhost.cpp "
fi

echo $CPPFLAGS

clang++ $CPPFLAGS page_rank.cpp debug/utils_debug.cpp && ./a.out
