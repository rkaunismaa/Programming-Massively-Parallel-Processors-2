/*
 * =============================================================================
 *  Chapter 10: Parallel Reduction — Fig 10.6: Simple Sum Reduction
 * =============================================================================
 *  Book:      Programming Massively Parallel Processors (Kirk, Hwu & El Hajj,
 *             4th Edition, Morgan Kaufmann 2023)
 *  Figure:    10.6 — Simple sum reduction kernel (interleaved addressing)
 *  Purpose:   Naive parallel reduction tree using stride-doubling.
 *             Each thread owns position 2*threadIdx.x (even locations).
 *             High control divergence: only threads with threadIdx.x % stride == 0
 *             are active each iteration, wasting 31/32 resources by iteration 5.
 *  Hardware:  GTX 1050, sm_61 (Pascal) — set device 1
 * =============================================================================
 *
 *  KERNEL CONCEPT
 *  ---------------------------------------------------------------------------
 *  - Input array of N elements, launch N/2 threads (one block)
 *  - stride doubles each iteration: 1, 2, 4, 8, ...
 *  - if (threadIdx.x % stride == 0) input[i] += input[i + stride]
 *  - After log2(N) iterations, input[0] contains the total sum
 *  - Thread 0 writes final result to output[0]
 *
 *  LIMITATION: Single block only — max 1024 threads, so max 2048 elements.
 *  This is lifted in later kernels (multiblock, coarsened).
 * =============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../common/cuda_utils.cuh"

// =============================================================================
//  Fig 10.6 — Simple sum reduction kernel (interleaved, stride-doubling)
// =============================================================================
__global__ void simple_sum_reduction_kernel(float* input, float* output) {
    // Owner position: even locations only
    int i = 2 * threadIdx.x;

    // Stride doubles each iteration: 1, 2, 4, 8, ..., blockDim.x
    // Must go up to blockDim.x to merge both halves of the input
    for (int stride = 1; stride <= blockDim.x; stride *= 2) {
        if (threadIdx.x % stride == 0) {
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

    // Input size — must be ≤ 2048 for single-block reduction
    const int N = 2048;
    const int THREADS = N / 2;  // N/2 threads for N elements

    printf("\n=== Fig 10.6: Simple Sum Reduction (Interleaved, Stride-Doubling) ===\n");
    printf("Input size: %d elements\n", N);
    printf("Threads:    %d (one block)\n", THREADS);
    printf("Iterations: %d\n\n", (int)log2((float)N));

    // Host allocations
    float* h_input = (float*)malloc(N * sizeof(float));
    float* h_output = (float*)malloc(sizeof(float));
    float* h_input_copy = (float*)malloc(N * sizeof(float));

    // Initialize with random floats in [0, 1)
    srand(42);
    for (int i = 0; i < N; i++) {
        h_input[i] = (float)rand() / (float)RAND_MAX;
    }

    // Save original for validation
    memcpy(h_input_copy, h_input, N * sizeof(float));

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
    simple_sum_reduction_kernel<<<1, THREADS>>>(d_input, d_output);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Restore input (warm-up modified it)
    CHECK_CUDA(cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice));

    // Timed run
    gpu_timer timer;
    timer.start();
    simple_sum_reduction_kernel<<<1, THREADS>>>(d_input, d_output);
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
    free(h_input_copy);

    return passed ? 0 : 1;
}
