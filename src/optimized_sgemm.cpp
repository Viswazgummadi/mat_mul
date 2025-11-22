#include "../include/matrix_utils.h"
#include <immintrin.h>
#include <vector>
#include <algorithm>

// Optimized SGEMM
// Uses packing (copying submatrices to contiguous memory) and a micro-kernel.
// This mimics how BLAS libraries work.

// Micro-kernel: 8x8 block
// Computes C_micro += A_micro * B_micro
// A_micro is 8xK, B_micro is Kx8 (packed)
// But for simplicity, we'll assume we are doing a small block update.
// We'll use a 6x16 kernel for AVX2 (2 registers for A, 4 for B accumulation? No, 16 floats is 2 registers).
// Let's stick to a classic 8x8 kernel or similar.
// Registers: 16 YMM registers available in AVX2.
// We can keep 8x8 C block in registers? 64 floats = 256 bytes. 
// 8 YMM registers can hold 8x8 floats? No, 1 YMM = 8 floats.
// So 8 YMM registers = 8x8 block of C.
// We need some registers for A and B.
// So we can use 8-12 registers for C, and rest for A/B.
// Let's use 8x8 kernel. C is 8x8.
// We load 1 float from A, broadcast it, multiply by row of B (8 floats).
// Repeat for k.

void kernel_8x8(int k, const float* A, const float* B, float* C, int ldc) {
    __m256 c[8];
    for (int i = 0; i < 8; ++i) {
        c[i] = _mm256_loadu_ps(&C[i * ldc]);
    }

    for (int p = 0; p < k; ++p) {
        __m256 b_vec = _mm256_loadu_ps(&B[p * 8]);
        
        for (int i = 0; i < 8; ++i) {
            __m256 a_vec = _mm256_set1_ps(A[i * k + p]); // A is stored row-major? No, we pack A.
            // If A is packed as 8xK, then A[i*k + p] is correct if row-major.
            // But usually we pack A as Kx8 or something for better access.
            // Let's assume A is packed such that we access it sequentially or broadcast easily.
            // If A is packed column-major (or panel), we might just load.
            // Let's stick to simple packing: A is 8 rows, K columns.
            // We want to load A[i][p].
            
            c[i] = _mm256_fmadd_ps(a_vec, b_vec, c[i]);
        }
    }

    for (int i = 0; i < 8; ++i) {
        _mm256_storeu_ps(&C[i * ldc], c[i]);
    }
}

// Packing functions
void pack_A(int k, const float* A, int lda, float* A_packed) {
    // Pack 8 rows of A into contiguous memory
    // A_packed will be 8 * k
    // We store it such that we can access it easily in the kernel.
    // Actually, for the kernel above: `a_vec = _mm256_set1_ps(A[i * k + p])`
    // This implies A is stored row-major 8xK.
    for (int i = 0; i < 8; ++i) {
        for (int p = 0; p < k; ++p) {
            A_packed[i * k + p] = A[i * lda + p];
        }
    }
}

void pack_B(int k, const float* B, int ldb, float* B_packed) {
    // Pack 8 columns of B into contiguous memory
    // B_packed will be k * 8
    // We store it row-major Kx8 so we can load `b_vec` as contiguous 8 floats.
    for (int p = 0; p < k; ++p) {
        for (int j = 0; j < 8; ++j) {
            B_packed[p * 8 + j] = B[p * ldb + j];
        }
    }
}

void matmul_optimized_sgemm(const Matrix& A, const Matrix& B, Matrix& C, int m, int n, int p) {
    // Block sizes
    const int MC = 256; // Block size for M
    const int KC = 256; // Block size for K
    const int NC = 128; // Block size for N

    // Buffers for packing
    // We need to be thread-safe if we parallelize this later.
    // For now, single threaded logic, but we can allocate inside.
    // Or static thread_local.
    std::vector<float> A_packed(MC * KC);
    std::vector<float> B_packed(KC * NC);

    for (int j = 0; j < p; j += NC) {
        int jb = std::min(NC, p - j);
        for (int k = 0; k < n; k += KC) {
            int kb = std::min(KC, n - k);
            
            // Pack B (kb x jb) -> we only pack strips of 8 columns?
            // The standard GotoBLAS approach loops over j, then k, then i.
            // B is packed once for all i.
            
            for (int i = 0; i < m; i += MC) {
                int ib = std::min(MC, m - i);
                
                // Inner loops over micro-blocks (8x8)
                for (int jj = 0; jj < jb; jj += 8) {
                    // Pack B strip (kb x 8)
                    // We verify we have 8 columns. If not, we need edge handling.
                    if (jj + 8 > jb) {
                        // Fallback for edge cases or implement masked kernel
                        // For this demo, we just skip or use naive for edges.
                        // Let's use naive for the whole block if it doesn't fit.
                        // Or just ignore edges for "Optimized" demo and assume multiple of 8.
                        // We'll implement a simple fallback loop for the whole function if dimensions are not multiples of 8.
                        continue; 
                    }
                    pack_B(kb, &B[(k * p) + (j + jj)], p, &B_packed[0]);

                    for (int ii = 0; ii < ib; ii += 8) {
                        if (ii + 8 > ib) continue;
                        
                        // Pack A strip (8 x kb)
                        pack_A(kb, &A[(i + ii) * n + k], n, &A_packed[0]);

                        // Kernel
                        kernel_8x8(kb, &A_packed[0], &B_packed[0], &C[(i + ii) * p + (j + jj)], p);
                    }
                }
            }
        }
    }
    
    // Handle edges (naive fallback for remaining elements)
    // This is a "massive project" so we should handle it, but for brevity in the demo code:
    // We will just run a cleanup pass or ensure inputs are multiples of 8 in benchmark.
    // Let's add a cleanup pass using the naive implementation for the whole matrix?
    // No, that's slow.
    // We'll just say "Optimized (8x8 aligned)" and require multiples of 8 for peak perf.
    // But to be correct, we should handle edges.
    // Let's just leave it as is for the "Optimized" core, and maybe the benchmark will show 0 for edges if we don't handle them.
    // Actually, let's just call naive for the whole thing if not multiple of 8, to ensure correctness.
    if (m % 8 != 0 || n % 8 != 0 || p % 8 != 0) {
        // Fallback
        // We can't easily mix and match without complex logic.
        // So we'll just run naive on the whole thing if not aligned.
        // This is a valid strategy for a "demo" of the kernel.
        matmul_naive(A, B, C, m, n, p);
    }
}
