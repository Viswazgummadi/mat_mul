# Matrix Multiplication Optimization Project Walkthrough

## Overview
This project explored various techniques to optimize matrix multiplication on the CPU, aiming for maximum performance. We implemented a range of algorithms from naive $O(n^3)$ loops to high-performance SIMD and blocked approaches.

## Implementations
All source code is located in the `src/` directory.

1.  **Naive (`naive.cpp`)**: The baseline triple-loop implementation.
2.  **Loop Reordering (`loop_reorder.cpp`)**: Optimizes memory access patterns (I-K-J) for better cache locality.
3.  **Tiled (`tiled.cpp`)**: Uses blocking to fit data into L1/L2 cache.
4.  **SIMD (`simd.cpp`)**: Uses AVX2 intrinsics for vectorization (8 floats per cycle).
5.  **OpenMP (`parallel_omp.cpp`)**: Parallelizes the outer loop using OpenMP.
6.  **Multi-threaded (`parallel_threads.cpp`)**: Manually manages `std::thread` for parallel execution.
7.  **Strassen (`strassen.cpp`)**: Recursive divide-and-conquer algorithm ($O(n^{2.81})$).
8.  **Optimized SGEMM (`optimized_sgemm.cpp`)**: Combines packing, micro-kernels, and blocking (BLAS-style).

## Reports
Detailed documentation and analysis were generated:

-   **[README.md](file:///c:/Users/viswa/OneDrive/Desktop/harshitha/README.md)**: Instructions for building and running the project.
-   **[Report.md](file:///c:/Users/viswa/OneDrive/Desktop/harshitha/docs/Report.md)**: A comprehensive explanation of each optimization technique and theoretical performance analysis.
-   **[Report.tex](file:///c:/Users/viswa/OneDrive/Desktop/harshitha/docs/Report.tex)**: A LaTeX version of the report for formal presentation.

## Benchmarking
A benchmark harness (`benchmark/benchmark_harness.cpp`) was created to verify correctness and measure GFLOPS.
*Note: Full benchmarking was skipped due to environment-specific compilation issues, but the harness is ready to run on a compatible setup.*

## How to Run
If a compatible C++ compiler (GCC/MinGW with OpenMP and AVX2 support) is available:

1.  **Using CMake**:
    ```bash
    mkdir build
    cd build
    cmake -G "MinGW Makefiles" ..
    cmake --build .
    ./benchmark_runner.exe
    ```

2.  **Manual Compilation**:
    ```bash
    g++ -O3 -march=native -mavx2 -mfma -fopenmp -I include src/*.cpp benchmark/benchmark_harness.cpp -o benchmark_runner.exe
    ./benchmark_runner.exe
    ```
