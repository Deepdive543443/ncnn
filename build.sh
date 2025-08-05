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
cmake -DNCNN_XTHEADVECTOR=OFF -DNCNN_BUILD_TESTS=ON -DCMAKE_TOOLCHAIN_FILE=/home/qfeng10/project/ncnn/toolchains/riscv64-unknown-linux-gnu.toolchain.cmake -DCMAKE_BUILD_TYPE=release ..
make -j 12
popd