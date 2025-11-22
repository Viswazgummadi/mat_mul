#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <map>
#include <functional>
#include <iomanip>
#include "../include/matrix_utils.h"

// Forward declarations of matmul functions
void matmul_naive(const Matrix& A, const Matrix& B, Matrix& C, int m, int n, int p);
void matmul_loop_reorder(const Matrix& A, const Matrix& B, Matrix& C, int m, int n, int p);
void matmul_tiled(const Matrix& A, const Matrix& B, Matrix& C, int m, int n, int p);
void matmul_simd(const Matrix& A, const Matrix& B, Matrix& C, int m, int n, int p);
void matmul_parallel_omp(const Matrix& A, const Matrix& B, Matrix& C, int m, int n, int p);
void matmul_parallel_threads(const Matrix& A, const Matrix& B, Matrix& C, int m, int n, int p);
void matmul_strassen(const Matrix& A, const Matrix& B, Matrix& C, int m, int n, int p);
    Matrix A, B, C_ref, C_test;
    randomize_matrix(A, m, n);
    randomize_matrix(B, n, p);
    zeros_matrix(C_ref, m, p);
    zeros_matrix(C_test, m, p);

    // Run reference implementation (Naive)
    bool verify = (m <= 512); 
    
    if (verify) {
        std::cout << "Generating reference result using Naive..." << std::endl;
        matmul_naive(A, B, C_ref, m, n, p);
    }

    std::cout << std::left << std::setw(25) << "Method" 
              << std::setw(15) << "Time (s)" 
              << std::setw(15) << "GFLOPS" 
              << "Status" << std::endl;
    std::cout << std::string(65, '-') << std::endl;

    for (const auto& [name, func] : methods) {
        double min_time = 1e9;
        
        // Warmup
        if (iterations > 1) func(A, B, C_test, m, n, p);

        for (int iter = 0; iter < iterations; ++iter) {
            zeros_matrix(C_test, m, p); // Reset result
            auto start = std::chrono::high_resolution_clock::now();
            func(A, B, C_test, m, n, p);
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> diff = end - start;
            if (diff.count() < min_time) min_time = diff.count();
        }

        // Calculate GFLOPS: 2*m*n*p floating point operations
        double gflops = (2.0 * m * n * p) / (min_time * 1e9);
        
        std::string status = "N/A";
        if (verify) {
            if (verify_matrix(C_ref, C_test)) {
                status = "PASS";
            } else {
                status = "FAIL";
            }
        }

        std::cout << std::left << std::setw(25) << name 
                  << std::setw(15) << std::fixed << std::setprecision(4) << min_time 
                  << std::setw(15) << std::fixed << std::setprecision(2) << gflops 
                  << status << std::endl;
    }
    std::cout << std::endl;
}

int main(int argc, char** argv) {
    // Default sizes
    std::vector<std::tuple<int, int, int>> sizes = {
        {128, 128, 128},
        {256, 256, 256},
        {512, 512, 512},
        {1024, 1024, 1024} 
    };

    for (const auto& [m, n, p] : sizes) {
        run_benchmark(m, n, p, 3);
    }

    return 0;
}
