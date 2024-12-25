#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <chrono> // Include chrono library for timing

typedef struct {
    float* Matrix1;
    float* Matrix2;
    float* dest;
    int DIM_R1;
    int DIM_C1;
    int DIM_C2;
    int blockSize;
    int startRow;
    int endRow;
} ThreadData;

// Function to perform blocked matrix multiplication for a subset of rows
void* matrixMultiplyBlockedThread(void* arg) {
    ThreadData* data = (ThreadData*)arg;
    float* Matrix1 = data->Matrix1;
    float* Matrix2 = data->Matrix2;
    float* dest = data->dest;
    int DIM_R1 = data->DIM_R1;
    int DIM_C1 = data->DIM_C1;
    int DIM_C2 = data->DIM_C2;
    int b = data->blockSize;
    int startRow = data->startRow;
    int endRow = data->endRow;

    // Perform blocked matrix multiplication for the assigned rows
    for (int ii = startRow; ii < endRow; ii += b) {
        for (int jj = 0; jj < DIM_C2; jj += b) {
            for (int kk = 0; kk < DIM_C1; kk += b) {
                for (int i = ii; i < ii + b && i < endRow; ++i) {
                    for (int j = jj; j < jj + b && j < DIM_C2; ++j) {
                        float sum = 0.0f; // Initialize accumulator to zero

                        for (int k = kk; k < kk + b && k < DIM_C1; ++k) {
                            // Perform multiplication and accumulate
                            sum += Matrix1[i * DIM_C1 + k] * Matrix2[k * DIM_C2 + j];
                        }

                        // Add the accumulated sum to the destination matrix
                        dest[i * DIM_C2 + j] += sum;
                    }
                }
            }
        }
    }

    pthread_exit(NULL);
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
    const int numThreads = 8; // Number of threads

    // Allocate and initialize matrices
    float* Matrix1 = (float*)malloc(DIM_R1 * DIM_C1 * sizeof(float));
    float* Matrix2 = (float*)malloc(DIM_C1 * DIM_C2 * sizeof(float));
    float* dest = (float*)malloc(DIM_R1 * DIM_C2 * sizeof(float));

    srand(time(NULL));
    for (int i = 0; i < DIM_R1 * DIM_C1; ++i) Matrix1[i] = (float)(rand());
    for (int i = 0; i < DIM_C1 * DIM_C2; ++i) Matrix2[i] = (float)(rand());

    // Initialize destination matrix to zero
    for (int i = 0; i < DIM_R1 * DIM_C2; ++i) dest[i] = 0.0f;

    // Print input matrices
    //printf("Matrix1:\n");
    //printMatrix(Matrix1, DIM_R1, DIM_C1);

    //printf("\nMatrix2:\n");
    //printMatrix(Matrix2, DIM_C1, DIM_C2);

    // Create threads for parallel computation
    pthread_t threads[numThreads];
    ThreadData threadData[numThreads];
    int rowsPerThread = DIM_R1 / numThreads;

    // Measure execution time using chrono
    auto start = std::chrono::high_resolution_clock::now();

    for (int t = 0; t < numThreads; ++t) {
        threadData[t].Matrix1 = Matrix1;
        threadData[t].Matrix2 = Matrix2;
        threadData[t].dest = dest;
        threadData[t].DIM_R1 = DIM_R1;
        threadData[t].DIM_C1 = DIM_C1;
        threadData[t].DIM_C2 = DIM_C2;
        threadData[t].blockSize = blockSize;
        threadData[t].startRow = t * rowsPerThread;
        threadData[t].endRow = (t == numThreads - 1) ? DIM_R1 : (t + 1) * rowsPerThread;

        pthread_create(&threads[t], NULL, matrixMultiplyBlockedThread, (void*)&threadData[t]);
    }

    // Wait for all threads to finish
    for (int t = 0; t < numThreads; ++t) {
        pthread_join(threads[t], NULL);
    }

    auto end = std::chrono::high_resolution_clock::now();

    // Print the result matrix
    //printf("\nResult Matrix (dest):\n");
    //printMatrix(dest, DIM_R1, DIM_C2);

    // Print execution time
    std::chrono::duration<double> duration = end - start;
    printf("\nExecution Time: %.6f\n", duration.count());

    // Free allocated memory
    free(Matrix1);
    free(Matrix2);
    free(dest);

}
