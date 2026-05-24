#!/usr/bin/env bash

# Xuantie Toolchain (Used for C90X)
export RISCV_ROOT_PATH=~/toolchain/Xuantie-900-gcc-linux-6.6.36-glibc-x86_64-V3.3.0

##### avaota-f2-riscv64-vlen256
mkdir -p build-avaota-f2
pushd build-avaota-f2
cmake -DCMAKE_TOOLCHAIN_FILE=../toolchains/c907-rv32-v310.toolchain.cmake \
        -DNCNN_BUILD_TESTS=ON \
        -DNCNN_BUILD_EXAMPLES=ON \
        -DNCNN_VULKAN=OFF \
        -DNCNN_BUILD_TOOLS=ON \
        ..
make -j12 install
popd

adb.exe devices
adb.exe shell mkdir -p /dev/ncnn
adb.exe push ./build-avaota-f2/benchmark/benchncnn /dev/ncnn
adb.exe shell chmod u+x /dev/ncnn/benchncnn
adb.exe shell /dev/ncnn/benchncnn
# rsync -avz ./build-avaota-f2 ./cmake ./tests ./examples ./benchmark ${USERNAME}@${REMOTE_HOST}:~/project/ncnn/
