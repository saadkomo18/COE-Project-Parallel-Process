#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <immintrin.h> // Include AVX header
#include <chrono> // For measuring execution time

// Function to perform blocked matrix multiplication with AVX
void matrixMultiplyBlockedAVX(float* Matrix1, float* Matrix2, float* dest, int DIM_R1, int DIM_C1, int DIM_C2, int b) {
    // Initialize the result matrix `dest` to zero
    for (int i = 0; i < DIM_R1; i += b) {
        for (int j = 0; j < DIM_C2; j += b) {
            for (int ii = i; ii < i + b && ii < DIM_R1; ++ii) {
                for (int jj = j; jj < j + b && jj < DIM_C2; ++jj) {
                    dest[ii * DIM_C2 + jj] = 0.0f;
                }
            }
        }
    }

    // Perform blocked matrix multiplication with AVX
    for (int ii = 0; ii < DIM_R1; ii += b) {
        for (int jj = 0; jj < DIM_C2; jj += b) {
            for (int kk = 0; kk < DIM_C1; kk += b) {
                for (int i = ii; i < ii + b && i < DIM_R1; ++i) {
                    for (int j = jj; j < jj + b && j < DIM_C2; ++j) {
                        __m256 sumVec = _mm256_setzero_ps(); // Initialize AVX vector accumulator to zero

                        for (int k = kk; k < kk + b && k < DIM_C1; k += 8) {
                            // Load 8 floats from Matrix1 and Matrix2
                            __m256 vecA = _mm256_loadu_ps(&Matrix1[i * DIM_C1 + k]);
                            __m256 vecB = _mm256_loadu_ps(&Matrix2[k * DIM_C2 + j]);

                            // Perform element-wise multiplication and accumulate
                            sumVec = _mm256_add_ps(sumVec, _mm256_mul_ps(vecA, vecB));
                        }

                        // Reduce the AVX vector to a scalar
                        float temp[8] = {0};
                        _mm256_storeu_ps(temp, sumVec);
                        float scalarSum = 0.0f;
                        for (int v = 0; v < 8; ++v) {
                            scalarSum += temp[v];
                        }

                        // Add the scalar sum to the destination matrix
                        dest[i * DIM_C2 + j] += scalarSum;
                    }
                }
            }
        }
    }
}

// Function to print a matrix
void printMatrix(float* matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%.2f ", matrix[i * cols + j]);
        }
        printf("\n");
    }
}

int main() {
    const int DIM_R1 = 4000; // Rows of Matrix1
    const int DIM_C1 = 4000; // Columns of Matrix1 (and Rows of Matrix2)
    const int DIM_C2 = 4000; // Columns of Matrix2
    const int blockSize = 8; // Block size for sub-matrix multiplication

    // Allocate and initialize matrices
    float* Matrix1 = (float*)aligned_alloc(32, DIM_R1 * DIM_C1 * sizeof(float));
    float* Matrix2 = (float*)aligned_alloc(32, DIM_C1 * DIM_C2 * sizeof(float));
    float* dest = (float*)aligned_alloc(32, DIM_R1 * DIM_C2 * sizeof(float));

    srand(time(NULL));
    for (int i = 0; i < DIM_R1 * DIM_C1; ++i) Matrix1[i] = (float)(rand());
    for (int i = 0; i < DIM_C1 * DIM_C2; ++i) Matrix2[i] = (float)(rand());

    // Print input matrices
    //printf("Matrix1:\n");
    //printMatrix(Matrix1, DIM_R1, DIM_C1);
    //printf("\nMatrix2:\n");
    //printMatrix(Matrix2, DIM_C1, DIM_C2);

    // Measure execution time for blocked matrix multiplication with AVX
    auto start = std::chrono::high_resolution_clock::now();
    matrixMultiplyBlockedAVX(Matrix1, Matrix2, dest, DIM_R1, DIM_C1, DIM_C2, blockSize);
    auto end = std::chrono::high_resolution_clock::now();
    // Print execution time
    std::chrono::duration<double> duration = end - start;
    printf("\nExecution Time: %.6f seconds\n", duration.count());

    // Print the result matrix
    //printf("\nResult Matrix (dest):\n");
    //printMatrix(dest, DIM_R1, DIM_C2);

    // Free allocated memory
    free(Matrix1);
    free(Matrix2);
    free(dest);

    return 0;
}

