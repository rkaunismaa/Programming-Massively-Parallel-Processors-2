/*
 * =============================================================================
 *  Chapter 10: Parallel Reduction — Fig 10.9: Convergent Reduction
 * =============================================================================
 *  Book:      Programming Massively Parallel Processors (Kirk, Hwu & El Hajj,
 *             4th Edition, Morgan Kaufmann 2023)
 *  Figure:    10.9 — Convergent reduction kernel (stride-halving)
 *  Purpose:   Improved reduction with reduced control divergence.
 *             Owner positions are consecutive (threadIdx.x), stride halves
 *             each iteration. Adjacent threads in each warp take the same
 *             code path — no divergence until the final 5 iterations.
 *             Also provides memory coalescing (adjacent threads access
 *             adjacent memory).
 *  Hardware:  GTX 1050, sm_61 (Pascal) — set device 1
 * =============================================================================
 *
 *  KERNEL CONCEPT
 *  ---------------------------------------------------------------------------
 *  - Input array of 2*blockDim.x elements, launch blockDim.x threads
 *  - Owner position: threadIdx.x (consecutive — adjacent threads are neighbors)
 *  - stride halves: blockDim.x, blockDim.x/2, ..., 1
 *  - if (threadIdx.x < stride) input[i] += input[i + stride]
 *  - Entire warps become inactive (no divergence within active warps)
 *  - ~66% resource utilization (vs ~35% for Fig 10.6)
 *  - 3.9× fewer global memory requests than Fig 10.6
 * =============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../common/cuda_utils.cuh"

// =============================================================================
//  Fig 10.9 — Convergent reduction kernel (stride-halving)
// =============================================================================
__global__ void convergent_reduction_kernel(float* input, float* output) {
    // Owner position: consecutive (adjacent threads own adjacent data)
    int i = threadIdx.x;

    // Stride halves each iteration: blockDim.x, blockDim.x/2, ..., 1
    for (int stride = blockDim.x; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            input[i] += input[i + stride];
        }
        __syncthreads();
    }

    // Thread 0 writes final result
    if (threadIdx.x == 0) {
        output[0] = input[0];
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

    // Input size — single block, so N = 2 * blockDim.x ≤ 2048
    const int BLOCK_SIZE = 1024;
    const int N = 2 * BLOCK_SIZE;

    printf("\n=== Fig 10.9: Convergent Reduction (Stride-Halving) ===\n");
    printf("Input size: %d elements\n", N);
    printf("Threads:    %d (one block)\n", BLOCK_SIZE);
    printf("Iterations: %d\n\n", (int)log2((float)N));

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
    convergent_reduction_kernel<<<1, BLOCK_SIZE>>>(d_input, d_output);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Restore input (warm-up modified it in-place)
    CHECK_CUDA(cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice));

    // Timed run
    gpu_timer timer;
    timer.start();
    convergent_reduction_kernel<<<1, BLOCK_SIZE>>>(d_input, d_output);
    timer.stop();
    float elapsed = timer.elapsed_ms();

    // Copy result back
    CHECK_CUDA(cudaMemcpy(h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost));

    // Validation
    printf("GPU sum: %.6f\n", h_output[0]);
    float diff = fabsf(h_output[0] - cpu_result);
    bool passed = diff < 1e-4f * fmaxf(1.0f, cpu_result);
    printf("Difference: %.6e\n", diff);

    printf("\n---------------------------------------------------------\n");
    printf("Kernel time: %.3f ms\n", elapsed);
    printf("Bandwidth:   %.2f GB/s\n",
           (2.0f * N * sizeof(float)) / (elapsed * 1e6f));
    printf("Validation:  %s\n", passed ? "PASSED" : "FAILED");
    printf("---------------------------------------------------------\n");

    // Cleanup
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));
    free(h_input);
    free(h_output);

    return passed ? 0 : 1;
}
