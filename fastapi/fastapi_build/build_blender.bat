@echo off
setlocal

:: Blender Build Folder (upper fastapi_build)
set BUILD_DIR=..\build_blender
set CONFIG=Release

:: Generator Visual Studio
set GENERATOR="Visual Studio 17 2022"
set ARCH=x64

:: Path to Blender source (one level upper fastapi_build)
set BLENDER_ROOT=..

:: Check and create build folder
if not exist %BUILD_DIR% mkdir %BUILD_DIR%

cd %BUILD_DIR%

echo [CMake] Configuring Blender with override config...
cmake -G %GENERATOR% -A %ARCH% -C ..\fastapi_build\blender_config_override.cmake %BLENDER_ROOT%

echo [CMake] Building Blender (%CONFIG%)...
cmake --build . --config %CONFIG%

cd ..
echo [Done]
pause