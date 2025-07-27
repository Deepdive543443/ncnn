#!/usr/bin/env bash

##### linux host system with gcc/g++
mkdir -p build
pushd build
cmake -DCMAKE_TOOLCHAIN_FILE=../toolchains/host.gcc.toolchain.cmake -DNCNN_BENCHMARK=ON ..
make -j6
cp benchmark/benchncnn ../benchmark