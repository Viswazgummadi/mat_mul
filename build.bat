@echo off
if not exist build mkdir build
"C:\MinGW\bin\g++.exe" -O3 -march=native -mavx2 -mfma -fopenmp -I include src/naive.cpp src/loop_reorder.cpp src/tiled.cpp src/simd.cpp src/parallel_omp.cpp src/parallel_threads.cpp src/strassen.cpp src/optimized_sgemm.cpp benchmark/benchmark_harness.cpp -o build/benchmark_runner.exe
if %errorlevel% neq 0 (
    echo Compilation failed!
    exit /b %errorlevel%
)
echo Compilation successful!
build\benchmark_runner.exe
