#!/usr/bin/env bash

USERNAME=$1
REMOTE_HOST=$2

##### Build riscv64
export RISCV_ROOT_PATH=~/toolchain/riscv64-glibc-ubuntu-22.04-gcc

mkdir -p build-riscv64
pushd build-riscv64
cmake -DCMAKE_TOOLCHAIN_FILE=../toolchains/riscv64-unknown-linux-gnu.toolchain.cmake \
        -DNCNN_BUILD_TESTS=ON \
        -DNCNN_BUILD_EXAMPLES=ON \
        -DNCNN_VULKAN=OFF \
        -DNCNN_BUILD_TOOLS=ON \
        ..

make -j32
make install
popd

##### Upload and test on remote riscv64 SBC
# ssh ${USERNAME}@${REMOTE_HOST} "mkdir -p ~/project/ncnn"
rsync -avz ./build-riscv64 ./cmake ./tests ./examples ./benchmark ${USERNAME}@${REMOTE_HOST}:~/project/ncnn/

# ssh ${USERNAME}@${REMOTE_HOST} "ctest --test-dir ~/project/ncnn/build-riscv64 --output-on-failure"

# ssh ${USERNAME}@${REMOTE_HOST} " \
#         ~/project/ncnn/build-riscv64/benchmark/benchncnn 4 8 2 -1 1 param=~/project/ncnn/benchmark/mobilenet.param shape=[224,224,3]; \
#         ~/project/ncnn/build-riscv64/benchmark/benchncnn 4 8 2 -1 1 param=~/project/ncnn/benchmark/mobilenet_int8.param shape=[224,224,3]"
