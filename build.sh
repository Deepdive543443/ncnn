#!/usr/bin/env bash

##### linux host system with gcc/g++
mkdir -p build_asan
pushd build_asan
cmake -DCMAKE_TOOLCHAIN_FILE=../toolchains/host.gcc.toolchain.cmake -DNCNN_BENCHMARK=ON ..
make -j6
cp benchmark/benchncnn ../benchmark