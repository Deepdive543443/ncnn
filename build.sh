#!/usr/bin/env bash

##### linux host system with gcc/g++
# mkdir -p build
# pushd build
# cmake -DCMAKE_TOOLCHAIN_FILE=../toolchains/host.gcc.toolchain.cmake -DNCNN_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=release ..
# make -j 12
# ctest -j 12
# popd

export RISCV_ROOT_PATH=~/toolchains/riscv64-gcc-15-glibc-nightly/

mkdir -p build
pushd build
cmake -DNCNN_XTHEADVECTOR=OFF -DNCNN_BUILD_TESTS=ON -DNCNN_BUILD_TOOLS=ON -DNCNN_BUILD_EXAMPLES=ON -DCMAKE_TOOLCHAIN_FILE=/home/qfeng10/project/ncnn/toolchains/riscv64-unknown-linux-gnu.toolchain.cmake -DCMAKE_BUILD_TYPE=release ..
make -j 12
popd


export RISCV_ROOT_PATH=~/toolchains/Xuantie-900-gcc-linux-6.6.0-glibc-x86_64-V3.2.0/
mkdir -p build_thead
pushd build_thead
cmake -DCMAKE_TOOLCHAIN_FILE=../toolchains/c910-v310.toolchain.cmake -DCMAKE_BUILD_TYPE=release \
      -DNCNN_OPENMP=ON -DNCNN_THREADS=ON \
      -DNCNN_RUNTIME_CPU=OFF \
      -DNCNN_RVV=OFF \
      -DNCNN_XTHEADVECTOR=ON \
      -DNCNN_ZFH=ON \
      -DNCNN_ZVFH=OFF \
      -DNCNN_SIMPLEOCV=ON -DNCNN_BUILD_EXAMPLES=ON -DNCNN_BUILD_TESTS=ON ..
make -j 12
popd
