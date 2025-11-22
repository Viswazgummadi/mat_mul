# Matrix Multiplication Optimization Report

## 1. Introduction
Matrix multiplication ($C = A \times B$) is a fundamental operation in high-performance computing, deep learning, and scientific simulations. The naive algorithm has a time complexity of $O(n^3)$. However, modern hardware offers opportunities for significant speedups through cache optimization, vectorization (SIMD), and parallelization.

This report details the implementation and theoretical performance of various optimization techniques.

## 2. Implementations

### 2.1 Naive Implementation
The standard implementation uses three nested loops (i, j, k).
```cpp
for (i=0; i<m; i++)
  for (j=0; j<p; j++)
    for (k=0; k<n; k++)
      C[i][j] += A[i][k] * B[k][j];
```
**Analysis**:
- **Memory Access**: `A` is accessed sequentially (good), but `B` is accessed with a stride of `p` (bad). This leads to frequent cache misses.
- **Performance**: Poor, typically < 5% of peak CPU performance.

### 2.2 Loop Reordering (Cache Optimization)
By changing the loop order to i-k-j, we access `B` sequentially.
```cpp
for (i=0; i<m; i++)
  for (k=0; k<n; k++)
    for (j=0; j<p; j++)
      C[i][j] += A[i][k] * B[k][j];
```
**Analysis**:
- **Memory Access**: `B[k][j]` is accessed sequentially. `C[i][j]` is also accessed sequentially. `A[i][k]` is constant for the inner loop.
- **Performance**: Significant improvement (often 5-10x faster than naive) due to spatial locality.

### 2.3 Tiled / Blocked Implementation
Tiling divides the matrices into small blocks that fit into the CPU's L1/L2 cache.
**Analysis**:
- **Cache Reuse**: Elements in a block are reused multiple times while they stay in the cache.
- **Performance**: Essential for large matrices where data doesn't fit in cache. Reduces memory bandwidth pressure.

### 2.4 Vectorization (SIMD)
Single Instruction, Multiple Data (SIMD) instructions (AVX/AVX2) allow the CPU to perform multiple operations (e.g., 8 floats) in a single clock cycle.
**Analysis**:
- **Throughput**: Theoretically increases peak performance by 8x (for AVX2 float).
- **Implementation**: Uses intrinsics like `_mm256_fmadd_ps`. Requires data alignment for best performance.

### 2.5 Parallelization (OpenMP / Threads)
Utilizes multi-core processors to compute different parts of the result matrix in parallel.
**Analysis**:
- **Scalability**: Performance scales almost linearly with the number of cores (up to memory bandwidth limits).
- **OpenMP**: Simple `#pragma omp parallel for` directives.
- **std::thread**: Manual thread management for fine-grained control.

### 2.6 Strassen's Algorithm
A divide-and-conquer algorithm with complexity $O(n^{\log_2 7}) \approx O(n^{2.81})$.
**Analysis**:
- **Theory**: Reduces the number of multiplications from 8 to 7 for each $2 \times 2$ block.
- **Practicality**: High overhead for small matrices. Effective for very large matrices ($N > 1000$). Often combined with standard algorithms for leaf nodes.

### 2.7 Optimized SGEMM (BLAS-style)
Combines all techniques:
1.  **Packing**: Copies data into contiguous buffers to minimize TLB misses and cache conflicts.
2.  **Micro-kernel**: A highly optimized, hand-tuned kernel (often in assembly) that computes a small block (e.g., 8x8) using full register file.
3.  **Blocking**: Multi-level blocking for L1, L2, L3 caches.
**Analysis**:
- **Performance**: Can achieve > 90% of theoretical peak CPU performance. This is how libraries like OpenBLAS and Intel MKL work.

## 3. Performance Metrics (Theoretical)
Assuming a modern CPU (e.g., 4GHz, 8 cores, AVX2):
- **Peak GFLOPS** = $4 \text{ GHz} \times 8 \text{ cores} \times 16 \text{ FLOPS/cycle} \approx 512 \text{ GFLOPS}$.

| Method | Estimated Efficiency | Estimated GFLOPS |
| :--- | :--- | :--- |
| Naive | < 1% | ~2-5 |
| Loop Reorder | 5-10% | ~20-50 |
| Tiled | 10-20% | ~50-100 |
| SIMD + Tiled | 40-60% | ~200-300 |
| Optimized SGEMM | 80-90% | ~400+ |

## 4. Conclusion
To achieve "super fast" matrix multiplication on the CPU, one must exploit the entire memory hierarchy and vector units. The Optimized SGEMM approach, combining packing, micro-kernels, and multi-threading, is the state-of-the-art method for dense matrix multiplication.
