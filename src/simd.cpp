#include "../include/matrix_utils.h"
#include <immintrin.h>

// SIMD Implementation using AVX2
// Processes 8 floats at a time.
void matmul_simd(const Matrix& A, const Matrix& B, Matrix& C, int m, int n, int p) {
    // We assume C is zeroed.
    
    // For SIMD, it's best to use the I-K-J loop order so we can load a scalar from A
    // and multiply it by a vector from B, then add to a vector in C.
    // C[i][j...j+7] += A[i][k] * B[k][j...j+7]
    
    for (int i = 0; i < m; ++i) {
        for (int k = 0; k < n; ++k) {
            // Broadcast A[i][k] to a vector
            __m256 a_vec = _mm256_set1_ps(A[i * n + k]);
            
            for (int j = 0; j < p; j += 8) {
                // Check boundary
                if (j + 8 > p) {
                    // Fallback for remaining elements
                    float a_val = A[i * n + k];
                    for (int jj = j; jj < p; ++jj) {
                        C[i * p + jj] += a_val * B[k * p + jj];
                    }
                    break;
                }

                // Load C[i][j...j+7]
                __m256 c_vec = _mm256_loadu_ps(&C[i * p + j]);
                
                // Load B[k][j...j+7]
                __m256 b_vec = _mm256_loadu_ps(&B[k * p + j]);
                
                // FMA: c_vec = c_vec + a_vec * b_vec
                // _mm256_fmadd_ps is AVX2 (FMA3). If not available, use mul + add.
                // Assuming AVX2 is available on modern CPUs.
                c_vec = _mm256_fmadd_ps(a_vec, b_vec, c_vec);
                
                // Store back to C
                _mm256_storeu_ps(&C[i * p + j], c_vec);
            }
        }
    }
}
