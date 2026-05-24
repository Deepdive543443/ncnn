#!/usr/bin/env bash

USERNAME=$1
REMOTE_HOST=$2

# SpacemiT Toolchain (Used for x60 / K1)
export RISCV_ROOT_PATH=~/toolchain/spacemit-toolchain-linux-glibc-x86_64-v1.2.2

##### orangepirv2-riscv64-vlen256
mkdir -p build-orangepirv2
pushd build-orangepirv2
cmake -DCMAKE_TOOLCHAIN_FILE=../toolchains/k1.toolchain.cmake \
        -DNCNN_BUILD_TESTS=ON \
        -DNCNN_BUILD_EXAMPLES=ON \
        -DNCNN_VULKAN=OFF \
        -DNCNN_BUILD_TOOLS=ON \
        ..
make -j$(nproc)
make install
popd

rsync -avz ./build-orangepirv2 ./cmake ./tests ./examples ./benchmark ${USERNAME}@${REMOTE_HOST}:~/project/ncnn/
