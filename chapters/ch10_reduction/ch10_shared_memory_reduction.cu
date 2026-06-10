/*
 * =============================================================================
 *  Chapter 10: Parallel Reduction — Fig 10.11: Shared Memory Reduction
 * =============================================================================
 *  Book:      Programming Massively Parallel Processors (Kirk, Hwu & El Hajj,
 *             4th Edition, Morgan Kaufmann 2023)
 *  Figure:    10.11 — Shared memory reduction kernel
 *  Purpose:   Uses shared memory to drastically reduce global memory accesses.
 *             Each thread loads 2 elements from global memory, adds them,
 *             and stores the partial sum in shared memory. All subsequent
 *             iterations operate entirely in shared memory.
 *             Global memory accesses: only N+1 (N reads for initial load +
 *             1 write for final output) vs 3*N/2 for the global-memory version.
 *  Hardware:  GTX 1050, sm_61 (Pascal) — set device 1
 * =============================================================================
 *
 *  KERNEL CONCEPT
 *  ---------------------------------------------------------------------------
 *  - Each thread loads input[t] + input[t + blockDim.x] → shared[t]
 *  - This "chews up" the first iteration (pairwise sum) before the loop
 *  - Loop starts at blockDim.x/2 instead of blockDim.x
 *  - __syncthreads() at loop entry ensures shared memory ready
 *  - Thread 0 writes shared[0] to output
 *  - Input array is NOT modified (useful if original values needed later)
 * =============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../common/cuda_utils.cuh"

// =============================================================================
//  Fig 10.11 — Shared memory reduction kernel
// =============================================================================
__global__ void shared_memory_reduction_kernel(const float* input,
                                               float* output) {
    __shared__ float input_s[1024];
    int t = threadIdx.x;

    // First iteration: load from global, add pair, store to shared
    input_s[t] = input[t] + input[t + blockDim.x];
    __syncthreads();

    // Remaining iterations in shared memory (start at blockDim.x/2)
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (t < stride) {
            input_s[t] += input_s[t + stride];
        }
        __syncthreads();
    }

    // Thread 0 writes final result
    if (t == 0) {
        output[0] = input_s[0];
    }
}

// =============================================================================
//  CPU reference: sequential sum
// =============================================================================
float cpu_sum(const float* data, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += data[i];
    }
    return (float)sum;
}

// =============================================================================
//  Main
// =============================================================================
int main() {
    // Set GTX 1050 (device 1)
    int dev = 1;
    cudaSetDevice(dev);
    print_device_info(dev);

    // Input size — 2 * blockDim.x (single block)
    const int BLOCK_SIZE = 1024;
    const int N = 2 * BLOCK_SIZE;

    printf("\n=== Fig 10.11: Shared Memory Reduction ===\n");
    printf("Input size: %d elements\n", N);
    printf("Threads:    %d (one block)\n", BLOCK_SIZE);
    printf("Iterations (shared): %d\n\n", (int)log2((float)BLOCK_SIZE));

    // Host allocations
    float* h_input = (float*)malloc(N * sizeof(float));
    float* h_output = (float*)malloc(sizeof(float));

    // Initialize with random floats in [0, 1)
    srand(42);
    for (int i = 0; i < N; i++) {
        h_input[i] = (float)rand() / (float)RAND_MAX;
    }

    // CPU reference
    float cpu_result = cpu_sum(h_input, N);
    printf("CPU sum: %.6f\n", cpu_result);

    // Device allocations
    float *d_input, *d_output;
    CHECK_CUDA(cudaMalloc(&d_input, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output, sizeof(float)));

    // Copy to device
    CHECK_CUDA(cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice));

    // Warm-up
    shared_memory_reduction_kernel<<<1, BLOCK_SIZE>>>(d_input, d_output);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Timed run
    gpu_timer timer;
    timer.start();
    shared_memory_reduction_kernel<<<1, BLOCK_SIZE>>>(d_input, d_output);
    timer.stop();
    float elapsed = timer.elapsed_ms();

    // Copy result back
    CHECK_CUDA(cudaMemcpy(h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost));

    // Validation
    printf("GPU sum: %.6f\n", h_output[0]);
    float diff = fabsf(h_output[0] - cpu_result);
    bool passed = diff < 1e-4f * fmaxf(1.0f, cpu_result);
    printf("Difference: %.6e\n", diff);

    // Global memory access count
    int global_reads = N;  // N elements loaded from global
    int global_writes = 1; // 1 final output write
    printf("Global mem accesses: %d (vs ~%d for Fig 10.9 global-memory version)\n",
           global_reads + global_writes, 3 * N / 2);

    printf("\n---------------------------------------------------------\n");
    printf("Kernel time: %.3f ms\n", elapsed);
    printf("Bandwidth:   %.2f GB/s\n",
           (float)(N * sizeof(float)) / (elapsed * 1e6f));
    printf("Validation:  %s\n", passed ? "PASSED" : "FAILED");
    printf("---------------------------------------------------------\n");

    // Verify input array is unchanged
    float check_val;
    CHECK_CUDA(cudaMemcpy(&check_val, &d_input[0], sizeof(float), cudaMemcpyDeviceToHost));
    printf("Input[0] preserved: %s (%.6f == %.6f)\n",
           fabsf(check_val - h_input[0]) < 1e-6f ? "YES" : "NO",
           check_val, h_input[0]);

    // Cleanup
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));
    free(h_input);
    free(h_output);

    return passed ? 0 : 1;
}
