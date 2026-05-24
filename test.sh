#!/usr/bin/env bash
set -e

# ==============================================================================
# Configuration: Update these paths to point to your local toolchains and QEMU
# ==============================================================================

# Xuantie Toolchain (Used for c906, c910, c908, c907, c907-rv32)
export XUANTIE_ROOT_PATH=~/toolchain/Xuantie-900-gcc-linux-6.6.36-glibc-x86_64-V3.3.0
export QEMU_XUANTIE_BIN=~/toolchain/xuantie-qemu-x86_64-Ubuntu-20.04-V5.4.1/bin

# SpacemiT Toolchain (Used for x60 / K1)
export SPACEMIT_ROOT_PATH=~/toolchain/spacemit-toolchain-linux-glibc-x86_64-v1.2.2
export QEMU_SPACEMIT_BIN=~/toolchain/jdsk-qemu/bin

# Standard RISC-V GNU Toolchain (Used for gcc-rvv and clang-rvv)
export RISCV_GNU_ROOT_PATH=~/toolchain/riscv64-glibc-ubuntu-22.04-gcc
export QEMU_INSTALL_BIN=~/toolchain/qemu-install/bin

# Note: The 'gcc-riscv64' job assumes you have `g++-riscv64-linux-gnu` installed system-wide:
# sudo apt-get install g++-riscv64-linux-gnu

# ==============================================================================
# Helper Function for Build and Test
# ==============================================================================
build_and_test() {
    local build_dir=$1
    local toolchain=$2
    local extra_cmake_args=$3
    local qemu_bin_path=$4
    local qemu_exe=$5
    local qemu_args=$6

    echo ""
    echo "=========================================================================="
    echo "Building and Testing: $build_dir"
    echo "=========================================================================="
    echo ""

    mkdir -p "$build_dir"
    pushd "$build_dir"

    cmake -DCMAKE_TOOLCHAIN_FILE="../$toolchain" -DCMAKE_BUILD_TYPE=release \
        -DNCNN_BUILD_TOOLS=OFF -DNCNN_BUILD_EXAMPLES=OFF -DNCNN_BUILD_TESTS=ON \
        $extra_cmake_args ..

    make -j$(nproc)

    # Prepend custom QEMU bin path if provided
    local ORIGINAL_PATH=$PATH
    if [ -n "$qemu_bin_path" ]; then
        export PATH="$qemu_bin_path:$PATH"
    fi

    # Run ctest with QEMU integration
    TESTS_EXECUTABLE_LOADER="$qemu_exe" TESTS_EXECUTABLE_LOADER_ARGUMENTS="$qemu_args" ctest --output-on-failure -j$(nproc)

    export PATH=$ORIGINAL_PATH
    popd
}


# ==============================================================================
# 1. gcc-riscv64 (Standard Ubuntu GCC)
# ==============================================================================
build_and_test "build-gcc-riscv64" \
    "toolchains/riscv64-linux-gnu.toolchain.cmake" \
    "" \
    "" \
    "qemu-riscv64" \
    "-L;/usr/riscv64-linux-gnu"

# ==============================================================================
# 2. GCC RVV (linux-riscv64.yml)
# ==============================================================================
# export RISCV_ROOT_PATH=$RISCV_GNU_ROOT_PATH
# echo ""
# echo "=========================================================================="
# echo "Building and Testing: build-gcc-rvv"
# echo "=========================================================================="
# mkdir -p build-gcc-rvv && pushd build-gcc-rvv
# cmake -DCMAKE_TOOLCHAIN_FILE=../toolchains/riscv64-unknown-linux-gnu.toolchain.cmake -DCMAKE_BUILD_TYPE=release -DNCNN_BUILD_TOOLS=OFF -DNCNN_BUILD_EXAMPLES=OFF -DNCNN_BUILD_TESTS=ON ..
# make -j$(nproc)
# export PATH="$QEMU_INSTALL_BIN:$PATH"

# echo "--- gcc-rvv test-vlen256 ---"
# TESTS_EXECUTABLE_LOADER=qemu-riscv64 TESTS_EXECUTABLE_LOADER_ARGUMENTS="-cpu;rv64,v=true,zfh=true,zvfh=true,vlen=256,elen=64,vext_spec=v1.0;-L;$RISCV_ROOT_PATH/sysroot" ctest --output-on-failure -j$(nproc)

# echo "--- gcc-rvv test-vlen128 ---"
# TESTS_EXECUTABLE_LOADER=qemu-riscv64 TESTS_EXECUTABLE_LOADER_ARGUMENTS="-cpu;rv64,v=true,zfh=true,zvfh=true,vlen=128,elen=64,vext_spec=v1.0;-L;$RISCV_ROOT_PATH/sysroot" ctest --output-on-failure -j$(nproc)
# popd


# ==============================================================================
# 3. SpacemiT (linux-riscv64.yml)
# ==============================================================================
export RISCV_ROOT_PATH=$SPACEMIT_ROOT_PATH

# x60 GCC
build_and_test "build-spacemit-gcc-vlen256" \
    "toolchains/k1.toolchain.cmake" \
    "-DNCNN_OPENMP=ON -DNCNN_THREADS=ON -DNCNN_RUNTIME_CPU=OFF -DNCNN_RVV=ON -DNCNN_XTHEADVECTOR=OFF -DNCNN_ZFH=ON -DNCNN_ZVFH=ON" \
    "$QEMU_SPACEMIT_BIN" "qemu-riscv64" "-cpu;max,vlen=256,elen=64,vext_spec=v1.0;-L;$RISCV_ROOT_PATH/sysroot"

# x60 GCC
build_and_test "build-spacemit-gcc-vlen128" \
    "toolchains/k1.toolchain.cmake" \
    "-DNCNN_OPENMP=ON -DNCNN_THREADS=ON -DNCNN_RUNTIME_CPU=OFF -DNCNN_RVV=ON -DNCNN_XTHEADVECTOR=OFF -DNCNN_ZFH=ON -DNCNN_ZVFH=ON" \
    "$QEMU_SPACEMIT_BIN" "qemu-riscv64" "-cpu;max,vlen=128,elen=64,vext_spec=v1.0;-L;$RISCV_ROOT_PATH/sysroot"

# # x60 LLVM
# build_and_test "build-spacemit-llvm" \
#     "toolchains/k1.llvm.toolchain.cmake" \
#     "-DNCNN_OPENMP=ON -DNCNN_THREADS=ON -DNCNN_RUNTIME_CPU=OFF -DNCNN_RVV=ON -DNCNN_XTHEADVECTOR=OFF -DNCNN_ZFH=ON -DNCNN_ZVFH=ON" \
#     "$QEMU_SPACEMIT_BIN" "qemu-riscv64" "-cpu;max,vlen=256,elen=64,vext_spec=v1.0;-L;$RISCV_ROOT_PATH/sysroot"


# ==============================================================================
# 4. Xuantie (linux-riscv64.yml)
# ==============================================================================
export RISCV_ROOT_PATH=$XUANTIE_ROOT_PATH

# c906 (THead Vector)
build_and_test "build-xuantie-c906" \
    "toolchains/c906-v310.toolchain.cmake" \
    "-DNCNN_OPENMP=OFF -DNCNN_THREADS=OFF -DNCNN_RUNTIME_CPU=OFF -DNCNN_RVV=OFF -DNCNN_XTHEADVECTOR=ON -DNCNN_ZFH=ON -DNCNN_ZVFH=OFF" \
    "$QEMU_XUANTIE_BIN" "qemu-riscv64" "-cpu;c906fdv"

# c910 (THead Vector)
build_and_test "build-xuantie-c910" \
    "toolchains/c910-v310.toolchain.cmake" \
    "-DNCNN_OPENMP=ON -DNCNN_THREADS=ON -DNCNN_RUNTIME_CPU=OFF -DNCNN_RVV=OFF -DNCNN_XTHEADVECTOR=ON -DNCNN_ZFH=ON -DNCNN_ZVFH=OFF" \
    "$QEMU_XUANTIE_BIN" "qemu-riscv64" "-cpu;c910v"

# c908 (RVV 1.0)
build_and_test "build-xuantie-c908" \
    "toolchains/c908-v310.toolchain.cmake" \
    "-DNCNN_OPENMP=ON -DNCNN_THREADS=ON -DNCNN_RUNTIME_CPU=OFF -DNCNN_RVV=ON -DNCNN_XTHEADVECTOR=OFF -DNCNN_ZFH=ON -DNCNN_ZVFH=ON" \
    "$QEMU_XUANTIE_BIN" "qemu-riscv64" "-cpu;c908v"

# c907 (RVV 1.0)
build_and_test "build-xuantie-c907" \
    "toolchains/c907-v310.toolchain.cmake" \
    "-DNCNN_OPENMP=ON -DNCNN_THREADS=ON -DNCNN_RUNTIME_CPU=OFF -DNCNN_RVV=ON -DNCNN_XTHEADVECTOR=OFF -DNCNN_ZFH=ON -DNCNN_ZVFH=ON" \
    "$QEMU_XUANTIE_BIN" "qemu-riscv64" "-cpu;c907fdv-rv64"


# ==============================================================================
# 5. Clang RVV (linux-riscv64.yml)
# ==============================================================================
# export RISCV_ROOT_PATH=$RISCV_GNU_ROOT_PATH
# echo ""
# echo "=========================================================================="
# echo "Building and Testing: build-clang-rvv"
# echo "=========================================================================="
# mkdir -p build-clang-rvv && pushd build-clang-rvv
# cmake -DCMAKE_TOOLCHAIN_FILE=../toolchains/riscv64-unknown-linux-gnu.llvm-toolchain.cmake -DCMAKE_BUILD_TYPE=release -DNCNN_BUILD_TOOLS=OFF -DNCNN_BUILD_EXAMPLES=OFF -DNCNN_BUILD_TESTS=ON ..
# make -j$(nproc)
# export PATH="$QEMU_INSTALL_BIN:$PATH"

# echo "--- clang-rvv test-vlen256 ---"
# TESTS_EXECUTABLE_LOADER=qemu-riscv64 TESTS_EXECUTABLE_LOADER_ARGUMENTS="-cpu;rv64,v=true,zfh=true,zvfh=true,vlen=256,elen=64,vext_spec=v1.0;-L;$RISCV_ROOT_PATH/sysroot" ctest --output-on-failure -j$(nproc)

# echo "--- clang-rvv test-vlen128 ---"
# TESTS_EXECUTABLE_LOADER=qemu-riscv64 TESTS_EXECUTABLE_LOADER_ARGUMENTS="-cpu;rv64,v=true,zfh=true,zvfh=true,vlen=128,elen=64,vext_spec=v1.0;-L;$RISCV_ROOT_PATH/sysroot" ctest --output-on-failure -j$(nproc)
# popd


# ==============================================================================
# 6. Xuantie RV32 (linux-riscv32.yml)
# ==============================================================================
export RISCV_ROOT_PATH=$XUANTIE_ROOT_PATH

# c907-rv32 (RVV 1.0 32-bit)
build_and_test "build-xuantie-c907-rv32" \
    "toolchains/c907-rv32-v310.toolchain.cmake" \
    "-DNCNN_OPENMP=ON -DNCNN_THREADS=ON -DNCNN_RUNTIME_CPU=OFF -DNCNN_RVV=ON -DNCNN_XTHEADVECTOR=OFF -DNCNN_ZFH=ON -DNCNN_ZVFH=ON" \
    "$QEMU_XUANTIE_BIN" "qemu-riscv32" "-cpu;c907fdv-rv32"

echo ""
echo "All workflows completed successfully!"
