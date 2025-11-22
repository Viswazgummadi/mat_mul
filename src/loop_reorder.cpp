#include "../include/matrix_utils.h"

// Loop Reordering: I-K-J
// Accesses B sequentially in memory (B[k][j]) which is much better than I-J-K where B is accessed with stride P.
void matmul_loop_reorder(const Matrix& A, const Matrix& B, Matrix& C, int m, int n, int p) {
    // Initialize C to 0
    // In this version, we might accumulate, so we ensure it's zeroed first.
    // The harness zeroes it, but let's be safe or assume it's zeroed.
    // Actually, for I-K-J, we load A[i][k] and multiply it by the row B[k].
    // So we iterate over C[i][j] in the inner loop.
    
    // We assume C is zeroed by the caller (benchmark harness does this).
    
    for (int i = 0; i < m; ++i) {
        for (int k = 0; k < n; ++k) {
            float a_val = A[i * n + k];
            for (int j = 0; j < p; ++j) {
                C[i * p + j] += a_val * B[k * p + j];
            }
        }
    }
}
