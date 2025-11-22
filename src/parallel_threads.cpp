#include "../include/matrix_utils.h"
#include <thread>
#include <vector>

// Multi-threaded Implementation using std::thread
// Manually manages threads to parallelize the outer loop.
void matmul_parallel_threads(const Matrix& A, const Matrix& B, Matrix& C, int m, int n, int p) {
    // We assume C is zeroed.
    
    int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4; // Fallback

    std::vector<std::thread> threads;
    int chunk_size = m / num_threads;
    int remainder = m % num_threads;

    int start_row = 0;
    for (int t = 0; t < num_threads; ++t) {
        int rows_for_thread = chunk_size + (t < remainder ? 1 : 0);
        int end_row = start_row + rows_for_thread;

        if (rows_for_thread > 0) {
            threads.emplace_back([=, &A, &B, &C]() {
                for (int i = start_row; i < end_row; ++i) {
                    for (int k = 0; k < n; ++k) {
                        float a_val = A[i * n + k];
                        for (int j = 0; j < p; ++j) {
                            C[i * p + j] += a_val * B[k * p + j];
                        }
                    }
                }
            });
        }
        start_row = end_row;
    }

    for (auto& th : threads) {
        if (th.joinable()) th.join();
    }
}
