#include "../include/matrix_utils.h"

// Naive I-J-K implementation
void matmul_naive(const Matrix& A, const Matrix& B, Matrix& C, int m, int n, int p) {
    // C is m x p
    // A is m x n
    // B is n x p
    
    // Ensure C is initialized to 0 if not already (though our benchmark harness will likely do it)
    // But strictly speaking, standard matmul accumulates, so we assume C is zeroed or we overwrite.
    // Here we overwrite.
    
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < p; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < n; ++k) {
                sum += A[i * n + k] * B[k * p + j];
            }
            C[i * p + j] = sum;
        }
    }
}
