#include "../include/matrix_utils.h"
#include <omp.h>

// OpenMP Implementation
// Parallelizes the outer loop using OpenMP.
void matmul_parallel_omp(const Matrix& A, const Matrix& B, Matrix& C, int m, int n, int p) {
    // We assume C is zeroed.
    
    // Simple parallelization of the I-K-J loop
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < m; ++i) {
        for (int k = 0; k < n; ++k) {
            float a_val = A[i * n + k];
            for (int j = 0; j < p; ++j) {
                C[i * p + j] += a_val * B[k * p + j];
            }
        }
    }
}
