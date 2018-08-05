#!/bin/bash

mkdir -p temp
cd temp

taco "y(i)=A(i,j)*x(j)" -f=y:d:0 -f=A:dd:0,1 -f=x:d:0 -write-source=taco_kernel.c -write-compute=taco_compute.c -write-assembly=taco_assembly.c

taco "y(i) = alpha * (A(i, j) * x(j)) + z(i)" \
-f=A:ss:0,1 \
-f=x:d:0 \
-f=z:d:0 \
-f=y:d:0 \
-t=A:double \
-t=x:double \
-t=z:double \
-t=y:double \
-write-source=page_rank_kernel.c \
-write-compute=page_rank_compute.c \
-write-assembly=page_rank_assembly.c

taco "x(i) = y(i)" \
-f=x:d:0 \
-f=y:d:0 \
-t=x:double \
-t=y:double \
-write-source=loop_kernel.c \
-write-compute=loop_compute.c \
-write-assembly=loop_assembly.c

cd ..