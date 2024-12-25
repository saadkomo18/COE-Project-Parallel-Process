#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <chrono> // Include chrono library for timing

// Function to perform blocked matrix multiplication (no AVX)
void matrixMultiplyBlocked(float* Matrix1, float* Matrix2, float* dest, int DIM_R1, int DIM_C1, int DIM_C2, int b) {
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

    // Perform blocked matrix multiplication
    for (int ii = 0; ii < DIM_R1; ii += b) {
        for (int jj = 0; jj < DIM_C2; jj += b) {
            for (int kk = 0; kk < DIM_C1; kk += b) {
                for (int i = ii; i < ii + b && i < DIM_R1; ++i) {
                    for (int j = jj; j < jj + b && j < DIM_C2; ++j) {
                        float val = dest[i * DIM_C2 + j];
                        for (int k = kk; k < kk + b && k < DIM_C1; ++k) {
                            val += Matrix1[i * DIM_C1 + k] * Matrix2[k * DIM_C2 + j];
                        }
                        dest[i * DIM_C2 + j] = val;
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
    float* Matrix1 = (float*)malloc(DIM_R1 * DIM_C1 * sizeof(float));
    float* Matrix2 = (float*)malloc(DIM_C1 * DIM_C2 * sizeof(float));
    float* dest = (float*)malloc(DIM_R1 * DIM_C2 * sizeof(float));

    srand(time(NULL));
    for (int i = 0; i < DIM_R1 * DIM_C1; ++i) Matrix1[i] = (float)(rand());
    for (int i = 0; i < DIM_C1 * DIM_C2; ++i) Matrix2[i] = (float)(rand());

    // Print input matrices
    //printf("Matrix1:\n");
    //printMatrix(Matrix1, DIM_R1, DIM_C1);
    //printf("\nMatrix2:\n");
    //printMatrix(Matrix2, DIM_C1, DIM_C2);

    // Measure execution time for blocked matrix multiplication
    auto start = std::chrono::high_resolution_clock::now();
    matrixMultiplyBlocked(Matrix1, Matrix2, dest, DIM_R1, DIM_C1, DIM_C2, blockSize);
    auto end = std::chrono::high_resolution_clock::now();
    // Print execution time
    std::chrono::duration<double> duration = end - start;
    printf("\nExecution Time: %.6f\n", duration.count());

    // Print the result matrix
    //printf("\nResult Matrix (dest):\n");
    //printMatrix(dest, DIM_R1, DIM_C2);

    // Free allocated memory
    free(Matrix1);
    free(Matrix2);
    free(dest);

    return 0;
}

