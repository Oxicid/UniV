@echo off
setlocal

set BUILD_DIR=build
set GENERATOR=Visual Studio 17 2022
set ARCH=x64
set CONFIG=Release


if not exist %BUILD_DIR% mkdir %BUILD_DIR%

cd %BUILD_DIR%


echo [CMake] Configuring project...
cmake -G "%GENERATOR%" -A %ARCH% ..


echo [CMake] Building project (%CONFIG%)...
cmake --build . --config %CONFIG%


cd ..
echo [Done]
pause