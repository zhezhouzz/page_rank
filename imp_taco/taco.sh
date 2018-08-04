#!/bin/bash

mkdir -p temp
cd temp

taco "PageMapHat(i,j) = PageMap(i,j) + d(i,j)" \
-f=PageMap:ss:0,1 \
-f=d:dd:0,1 \
-f=PageMapHat:dd:0,1 \
-t=PageMap:int32 \
-t=d:float \
-t=PageMapHat:float \
-write-source=prepare_kernel.c \
-write-compute=prepare_compute.c \
-write-assembly=prepare_assembly.c

taco "Rank(i) = PageMapHat(i,j) * RankLast(j)" \
-f=Rank:d:0 \
-f=PageMapHat:dd:0,1 \
-f=RankLast:d:0 \
-t=Rank:float \
-t=PageMapHat:float \
-t=RankLast:float \
-write-source=page_rank_kernel.c \
-write-compute=page_rank_compute.c \
-write-assembly=page_rank_assembly.c

taco "y(i)=A(i,j)*x(j)" -f=y:d:0 -f=A:dd:0,1 -f=x:d:0 -write-source=taco_kernel.c -write-compute=taco_compute.c -write-assembly=taco_assembly.c

cd ..