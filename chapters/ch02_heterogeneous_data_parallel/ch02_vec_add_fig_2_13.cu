// Book:      Programming Massively Parallel Processors (Kirk, Hwu & El Hajj, 4th ed.)
// Chapter:   2 — Heterogeneous Data Parallel Computing
// Reference: Figure 2.10 + Figure 2.13
// Concept:   Vector addition kernel with complete host wrapper — the "Hello World" of CUDA
// Key insight: Each thread handles one element; grid replaces the sequential for-loop
// Hardware:  GTX 1050, sm_61 (Pascal)
// Compile:   nvcc -std=c++17 -arch=sm_61 -O2 ch02_vec_add_fig_2_13.cu -o vec_add_fig_2_13

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <chrono>
#include "../common/cuda_utils.cuh"

// ============================================================
// Kernel: Figure 2.10 — vecAddKernel
// Each thread performs one pair-wise addition: C[i] = A[i] + B[i]
// ============================================================
__global__
void vecAddKernel(float* A, float* B, float* C, int n) {
    // Compute global thread index from block and thread indices
    // threadIdx.x: thread ID within block [0 .. blockDim.x-1]
    // blockIdx.x:  block ID within grid
    // blockDim.x: number of threads per block
    int i = threadIdx.x + blockDim.x * blockIdx.x;

    // Guard against threads beyond vector length (when n not multiple of block size)
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}

// ============================================================
// Host wrapper: Figure 2.13 — complete vecAdd function
// Allocates device memory, copies data, launches kernel, retrieves result
// ============================================================
void vecAdd(float* A_h, float* B_h, float* C_h, int n) {
    float *A_d, *B_d, *C_d;
    int size = n * sizeof(float);

    // Part 1: Allocate device global memory
    CHECK_CUDA(cudaMalloc((void**)&A_d, size));
    CHECK_CUDA(cudaMalloc((void**)&B_d, size));
    CHECK_CUDA(cudaMalloc((void**)&C_d, size));

    // Copy input vectors from host to device
    CHECK_CUDA(cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice));

    // Part 2: Launch kernel
    // Block size of 256 threads (multiple of warp size 32 for hardware efficiency)
    // Grid size = ceil(n / 256.0) blocks to cover all n elements
    int block_size = 256;
    int grid_size = (int)ceil(n / (float)block_size);
    vecAddKernel<<<grid_size, block_size>>>(A_d, B_d, C_d, n);
    CHECK_CUDA(cudaGetLastError()); // Check for kernel launch errors

    // Part 3: Copy result from device to host
    CHECK_CUDA(cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost));

    // Free device memory
    CHECK_CUDA(cudaFree(A_d));
    CHECK_CUDA(cudaFree(B_d));
    CHECK_CUDA(cudaFree(C_d));
}

// ============================================================
// Sequential reference for validation
// ============================================================
void vecAddRef(float* A, float* B, float* C, int n) {
    for (int i = 0; i < n; ++i) {
        C[i] = A[i] + B[i];
    }
}

// ============================================================
// Main: allocate, initialize, run GPU, validate against CPU
// ============================================================
int main() {
    // Print device info
    print_device_info();

    const int N = 1 << 20; // 1M elements (~4 MB per array)
    printf("Vector size: %d elements (%.2f MB per array)\n", N, (double)N * sizeof(float) / (1 << 20));

    // Allocate host memory
    float* A_h = (float*)malloc(N * sizeof(float));
    float* B_h = (float*)malloc(N * sizeof(float));
    float* C_h = (float*)malloc(N * sizeof(float));
    float* C_ref = (float*)malloc(N * sizeof(float));

    // Initialize with test data
    for (int i = 0; i < N; ++i) {
        A_h[i] = (float)(i % 100);
        B_h[i] = (float)((i * 7) % 100);
    }

    // Run GPU version with timing
    gpu_timer timer;
    timer.start();
    vecAdd(A_h, B_h, C_h, N);
    timer.stop();
    float gpu_ms = timer.elapsed_ms();

    // Run CPU reference
    vecAddRef(A_h, B_h, C_ref, N);

    // Validate GPU output against CPU reference
    bool passed = cpu_allclose(C_h, C_ref, N, 1e-5f);

    // Compute effective throughput metrics
    double bytes_transferred = 3.0 * N * sizeof(float); // A, B in + C out
    double bandwidth_gb_s = (bytes_transferred / (1 << 30)) / (gpu_ms / 1000.0);
    double gflops = (2.0 * N / 1e9) / (gpu_ms / 1000.0); // 1 add + 1 load per element

    printf("\n--- Results ---\n");
    printf("GPU time:     %.3f ms\n", gpu_ms);
    printf("Bandwidth:    %.2f GB/s\n", bandwidth_gb_s);
    printf("GFLOPS:       %.2f\n", gflops);
    printf("Validation:   %s\n", passed ? "PASS" : "FAIL");

    // Cleanup
    free(A_h);
    free(B_h);
    free(C_h);
    free(C_ref);

    return passed ? 0 : 1;
}
