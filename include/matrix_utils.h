#ifndef MATRIX_UTILS_H
#define MATRIX_UTILS_H

#include <vector>
#include <random>
#include <iostream>
#include <chrono>
#include <string>
#include <functional>
#include <iomanip>

// Using float for SGEMM (Single Precision General Matrix Multiply)
// Flattened 1D vector for better cache locality: A[i * cols + j]
using Matrix = std::vector<float>;

struct MatrixDims {
    int rows;
    int cols;
};

// Function signature for all matrix multiplication implementations
// C = A * B
// A: m x n, B: n x p, C: m x p
using MatMulFunc = std::function<void(const Matrix&, const Matrix&, Matrix&, int, int, int)>;

// Utility functions
inline void randomize_matrix(Matrix& mat, int rows, int cols) {
    static std::mt19937 gen(42); // Fixed seed for reproducibility
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    mat.resize(rows * cols);
    for (auto& val : mat) {
        val = dist(gen);
    }
}

inline void zeros_matrix(Matrix& mat, int rows, int cols) {
    mat.assign(rows * cols, 0.0f);
}

inline bool verify_matrix(const Matrix& expected, const Matrix& actual, float tol = 1e-4f) {
    if (expected.size() != actual.size()) return false;
    for (size_t i = 0; i < expected.size(); ++i) {
        if (std::abs(expected[i] - actual[i]) > tol) {
            return false;
        }
    }
    return true;
}

inline void print_matrix_small(const Matrix& mat, int rows, int cols, const std::string& name) {
    std::cout << "Matrix " << name << " (" << rows << "x" << cols << "):" << std::endl;
    for (int i = 0; i < std::min(rows, 5); ++i) {
        for (int j = 0; j < std::min(cols, 5); ++j) {
            std::cout << std::setw(8) << std::setprecision(3) << mat[i * cols + j] << " ";
        }
        if (cols > 5) std::cout << "...";
        std::cout << std::endl;
    }
    if (rows > 5) std::cout << "..." << std::endl;
    std::cout << std::endl;
}

#endif // MATRIX_UTILS_H
