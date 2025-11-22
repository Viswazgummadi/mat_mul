# Matrix Multiplication Optimization Project

This project explores various techniques to optimize matrix multiplication on the CPU, ranging from a naive implementation to high-performance SIMD and multi-threaded approaches.

## Implementations

The following methods are implemented in `src/`:

1.  **Naive (`naive.cpp`)**: Standard triple-loop implementation ($O(n^3)$). Baseline for correctness and performance.
2.  **Loop Reordering (`loop_reorder.cpp`)**: Optimizes memory access patterns (I-K-J) to improve cache locality.
3.  **Tiled/Blocked (`tiled.cpp`)**: Uses blocking to fit working sets into L1/L2 cache, reducing cache misses.
4.  **SIMD (`simd.cpp`)**: Utilizes AVX2 intrinsics to process 8 floating-point numbers in parallel per instruction.
5.  **OpenMP (`parallel_omp.cpp`)**: Parallelizes the outer loop using OpenMP for multi-core execution.
6.  **Multi-threaded (`parallel_threads.cpp`)**: Manually manages `std::thread` for parallel execution.
7.  **Strassen (`strassen.cpp`)**: Recursive implementation of Strassen's algorithm ($O(n^{\log_2 7}) \approx O(n^{2.81})$).
8.  **Optimized SGEMM (`optimized_sgemm.cpp`)**: A high-performance kernel using packing and micro-kernels, mimicking BLAS libraries.

## Building and Running

### Prerequisites
- C++ Compiler with C++17 support (GCC, Clang, or MSVC)
- CMake (optional, but recommended)
- OpenMP support (usually included with GCC/Clang)
- CPU with AVX2/FMA support (most modern CPUs)

### Using CMake
```bash
mkdir build
cd build
cmake -G "MinGW Makefiles" ..
cmake --build .
./benchmark_runner.exe
```

### Manual Compilation (Windows)
Run the provided `build.bat` script:
```cmd
build.bat
```
Or manually:
```bash
g++ -O3 -march=native -mavx2 -mfma -fopenmp -I include src/*.cpp benchmark/benchmark_harness.cpp -o benchmark_runner.exe
./benchmark_runner.exe
```

## Benchmarking

The `benchmark_harness.cpp` runs each method on matrices of varying sizes (128, 256, 512, 1024) and reports:
- Execution Time (seconds)
- Performance (GFLOPS)
- Correctness (PASS/FAIL vs Naive)

## Directory Structure
- `src/`: Source code for implementations.
- `include/`: Header files.
- `benchmark/`: Benchmarking harness.
- `docs/`: Detailed reports and analysis.
