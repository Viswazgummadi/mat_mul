#include "../include/matrix_utils.h"
#include <algorithm>

// Tiled / Blocked Implementation
// Uses blocking to keep working sets in L1/L2 cache.
void matmul_tiled(const Matrix& A, const Matrix& B, Matrix& C, int m, int n, int p) {
    // Block size should be tuned. 
    // L1 cache is usually 32KB. 3 matrices of block size BxB.
    // 3 * B^2 * 4 bytes <= 32KB roughly.
    // B^2 <= 2730 => B <= 52.
    // Let's pick 32 or 64. 32 is safe.
    const int BLOCK_SIZE = 32;

    // We assume C is zeroed.
    
    for (int i = 0; i < m; i += BLOCK_SIZE) {
        for (int k = 0; k < n; k += BLOCK_SIZE) {
            for (int j = 0; j < p; j += BLOCK_SIZE) {
                
                // Multiply the blocks
                int i_max = std::min(i + BLOCK_SIZE, m);
                int k_max = std::min(k + BLOCK_SIZE, n);
                int j_max = std::min(j + BLOCK_SIZE, p);

                for (int ii = i; ii < i_max; ++ii) {
                    for (int kk = k; kk < k_max; ++kk) {
                        float a_val = A[ii * n + kk];
                        for (int jj = j; jj < j_max; ++jj) {
                            C[ii * p + jj] += a_val * B[kk * p + jj];
                        }
                    }
                }
            }
        }
    }
}
