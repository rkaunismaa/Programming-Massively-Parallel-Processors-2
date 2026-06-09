// Book:      Programming Massively Parallel Processors (Kirk, Hwu & El Hajj, 4th ed.)
// Chapter:   2 — Heterogeneous Data Parallel Computing
// Reference: Exercise 2.2
// Concept:   Each thread processes TWO adjacent elements — strided access pattern
// Key insight: One thread handles elements [i] and [i+1], halving the required thread count
// Hardware:  GTX 1050, sm_61 (Pascal)
// Compile:   nvcc -std=c++17 -arch=sm_61 -O2 ex02.cu -o ex02

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include "../../common/cuda_utils.cuh"

// ============================================================
// Exercise 2.2: Each thread calculates TWO adjacent elements
// Index mapping: first element i = blockIdx.x * blockDim.x + threadIdx.x
//                second element i+1 (if within bounds)
// Answer: (C) i = (blockIdx.x * blockDim.x + threadIdx.x) — but thread handles [i, i+1]
// ============================================================
__global__
void vecAddTwoElements(float* A, float* B, float* C, int n) {
    // Compute global thread index
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // Each thread handles 2 elements: indices 2*tid and 2*tid+1
    int i = tid * 2;

    // Process first element
    if (i < n) {
        C[i] = A[i] + B[i];
    }

    // Process second adjacent element (if within bounds)
    if (i + 1 < n) {
        C[i + 1] = A[i + 1] + B[i + 1];
    }
}

void vecAddRef(float* A, float* B, float* C, int n) {
    for (int i = 0; i < n; ++i) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    print_device_info();

    const int N = 1000;
    printf("Vector size: %d elements\n", N);

    float* A_h = (float*)malloc(N * sizeof(float));
    float* B_h = (float*)malloc(N * sizeof(float));
    float* C_h = (float*)malloc(N * sizeof(float));
    float* C_ref = (float*)malloc(N * sizeof(float));

    for (int i = 0; i < N; ++i) {
        A_h[i] = (float)i;
        B_h[i] = (float)(i * 3);
    }

    // GPU version — need only half the threads since each handles 2 elements
    float *A_d, *B_d, *C_d;
    int size = N * sizeof(float);
    CHECK_CUDA(cudaMalloc((void**)&A_d, size));
    CHECK_CUDA(cudaMalloc((void**)&B_d, size));
    CHECK_CUDA(cudaMalloc((void**)&C_d, size));
    CHECK_CUDA(cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice));

    // Half the threads needed (each thread does 2 elements)
    int block_size = 256;
    int grid_size = (int)ceil(N / 2.0f / (float)block_size);
    printf("Grid: %d blocks x %d threads = %d threads (each handles 2 elements)\n",
           grid_size, block_size, grid_size * block_size);

    vecAddTwoElements<<<grid_size, block_size>>>(A_d, B_d, C_d, N);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost));

    // CPU reference
    vecAddRef(A_h, B_h, C_ref, N);

    bool passed = cpu_allclose(C_h, C_ref, N, 1e-5f);

    printf("\nExercise 2.2 — Two elements per thread\n");
    printf("Validation: %s\n", passed ? "PASS" : "FAIL");

    CHECK_CUDA(cudaFree(A_d));
    CHECK_CUDA(cudaFree(B_d));
    CHECK_CUDA(cudaFree(C_d));
    free(A_h); free(B_h); free(C_h); free(C_ref);

    return passed ? 0 : 1;
}
