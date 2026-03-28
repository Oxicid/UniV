#!/usr/bin/env bash

set -e

# Blender Build Folder
BUILD_DIR="../build_blender"
CONFIG="Release"

# Path to Blender source
BLENDER_ROOT=".."

# Create build dir if not exists
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

echo "[CMake] Configuring Blender with override config..."
cmake \
  -C ../fastapi_build/blender_config_override.cmake \
  -DCMAKE_BUILD_TYPE=$CONFIG \
  "$BLENDER_ROOT"

echo "[CMake] Building Blender..."
cmake --build . -j$(nproc)

cd ..
echo "[Done]"