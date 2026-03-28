#!/usr/bin/env bash

set -e

BUILD_DIR="build"
CONFIG="Release"

# Create build dir if not exists
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

echo "[CMake] Configuring project..."
cmake -DCMAKE_BUILD_TYPE=$CONFIG ..

echo "[CMake] Building project..."
cmake --build . -j$(nproc)

cd ..
echo "[Done]"