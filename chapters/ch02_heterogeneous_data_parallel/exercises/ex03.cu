// Book:      Programming Massively Parallel Processors (Kirk, Hwu & El Hajj, 4th ed.)
// Chapter:   2 — Heterogeneous Data Parallel Computing
// Reference: Exercise 2.3
// Concept:   Two-section processing pattern — threads process elements in two passes
// Key insight: Each block processes 2*blockDim.x elements in two sections, demonstrating
//              how threads can handle non-contiguous data with proper index mapping
// Hardware:  GTX 1050, sm_61 (Pascal)
// Compile:   nvcc -std=c++17 -arch=sm_61 -O2 ex03.cu -o ex03

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include "../../common/cuda_utils.cuh"

// ============================================================
// Exercise 2.3: Two-section processing pattern
// Each block processes 2*blockDim.x consecutive elements forming two sections:
//   Section 1: elements [base, base+blockDim.x-1]
//   Section 2: elements [base+blockDim.x, base+2*blockDim.x-1]
// Threads first process section 1, then section 2
// Index for first element: i = blockIdx.x * blockDim.x * 2 + threadIdx.x
// Index for second element: i + blockDim.x
// ============================================================
__global__
void vecAddTwoSections(float* A, float* B, float* C, int n) {
    // Each block handles 2*blockDim.x elements
    // Base index for this block's first section
    int base = blockIdx.x * blockDim.x * 2;
    int i = base + threadIdx.x;

    // Process first section element
    if (i < n) {
        C[i] = A[i] + B[i];
    }

    // Process second section element (offset by blockDim.x)
    int j = i + blockDim.x;
    if (j < n) {
        C[j] = A[j] + B[j];
    }
}

void vecAddRef(float* A, float* B, float* C, int n) {
    for (int i = 0; i < n; ++i) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    print_device_info();

    const int N = 2000;
    printf("Vector size: %d elements\n", N);

    float* A_h = (float*)malloc(N * sizeof(float));
    float* B_h = (float*)malloc(N * sizeof(float));
    float* C_h = (float*)malloc(N * sizeof(float));
    float* C_ref = (float*)malloc(N * sizeof(float));

    for (int i = 0; i < N; ++i) {
        A_h[i] = (float)i;
        B_h[i] = (float)(i * 5);
    }

    // GPU version
    float *A_d, *B_d, *C_d;
    int size = N * sizeof(float);
    CHECK_CUDA(cudaMalloc((void**)&A_d, size));
    CHECK_CUDA(cudaMalloc((void**)&B_d, size));
    CHECK_CUDA(cudaMalloc((void**)&C_d, size));
    CHECK_CUDA(cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice));

    // Each block handles 2*blockDim.x elements
    int block_size = 256;
    int grid_size = (int)ceil(N / (2.0f * block_size));
    printf("Grid: %d blocks x %d threads, each block handles %d elements\n",
           grid_size, block_size, 2 * block_size);

    vecAddTwoSections<<<grid_size, block_size>>>(A_d, B_d, C_d, N);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost));

    // CPU reference
    vecAddRef(A_h, B_h, C_ref, N);

    bool passed = cpu_allclose(C_h, C_ref, N, 1e-5f);

    printf("\nExercise 2.3 — Two-section processing\n");
    printf("Validation: %s\n", passed ? "PASS" : "FAIL");

    CHECK_CUDA(cudaFree(A_d));
    CHECK_CUDA(cudaFree(B_d));
    CHECK_CUDA(cudaFree(C_d));
    free(A_h); free(B_h); free(C_h); free(C_ref);

    return passed ? 0 : 1;
}
