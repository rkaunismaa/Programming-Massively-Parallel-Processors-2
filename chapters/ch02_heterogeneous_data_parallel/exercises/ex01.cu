// Book:      Programming Massively Parallel Processors (Kirk, Hwu & El Hajj, 4th ed.)
// Chapter:   2 — Heterogeneous Data Parallel Computing
// Reference: Exercise 2.1
// Concept:   Thread-to-data index mapping: i = blockIdx.x * blockDim.x + threadIdx.x
// Key insight: Global thread index combines block ID, block size, and thread ID within block
// Hardware:  GTX 1050, sm_61 (Pascal)
// Compile:   nvcc -std=c++17 -arch=sm_61 -O2 ex01.cu -o ex01

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include "../../common/cuda_utils.cuh"

// ============================================================
// Exercise 2.1: Verify the correct index mapping expression
// The correct answer is (C): i = blockIdx.x * blockDim.x + threadIdx.x
// This exercise demonstrates it works correctly across multiple blocks
// ============================================================
__global__
void vecAddKernel(float* A, float* B, float* C, int n) {
    // Correct mapping: global_index = block_id * threads_per_block + thread_id_in_block
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}

void vecAddRef(float* A, float* B, float* C, int n) {
    for (int i = 0; i < n; ++i) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    print_device_info();

    // Test with non-power-of-2 sizes to verify boundary handling
    const int N = 1000;
    printf("Vector size: %d elements (non-power-of-2 to test boundary)\n", N);

    float* A_h = (float*)malloc(N * sizeof(float));
    float* B_h = (float*)malloc(N * sizeof(float));
    float* C_h = (float*)malloc(N * sizeof(float));
    float* C_ref = (float*)malloc(N * sizeof(float));

    for (int i = 0; i < N; ++i) {
        A_h[i] = (float)i;
        B_h[i] = (float)(i * 2);
    }

    // GPU version
    float *A_d, *B_d, *C_d;
    int size = N * sizeof(float);
    CHECK_CUDA(cudaMalloc((void**)&A_d, size));
    CHECK_CUDA(cudaMalloc((void**)&B_d, size));
    CHECK_CUDA(cudaMalloc((void**)&C_d, size));
    CHECK_CUDA(cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice));

    int block_size = 256;
    int grid_size = (int)ceil(N / (float)block_size); // ceil(1000/256) = 4 blocks = 1024 threads
    printf("Grid: %d blocks x %d threads = %d total threads (n=%d)\n",
           grid_size, block_size, grid_size * block_size, N);

    vecAddKernel<<<grid_size, block_size>>>(A_d, B_d, C_d, N);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost));

    // CPU reference
    vecAddRef(A_h, B_h, C_ref, N);

    // Validate ALL elements
    bool passed = cpu_allclose(C_h, C_ref, N, 1e-5f);

    printf("\nExercise 2.1 — Index mapping verification\n");
    printf("Validation: %s\n", passed ? "PASS" : "FAIL");

    CHECK_CUDA(cudaFree(A_d));
    CHECK_CUDA(cudaFree(B_d));
    CHECK_CUDA(cudaFree(C_d));
    free(A_h); free(B_h); free(C_h); free(C_ref);

    return passed ? 0 : 1;
}
