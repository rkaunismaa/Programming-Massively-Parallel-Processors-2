/*
 * =============================================================================
 *  Chapter 10: Parallel Reduction — Fig 10.15: Thread-Coarsened Reduction
 * =============================================================================
 *  Book:      Programming Massively Parallel Processors (Kirk, Hwu & El Hajj,
 *             4th Edition, Morgan Kaufmann 2023)
 *  Figure:    10.15 — Sum reduction kernel with thread coarsening
 *  Purpose:   Reduces parallelization overhead by having each thread process
 *             COARSE_FACTOR elements before entering the reduction tree.
 *             Fewer blocks launched → less hardware underutilization during
 *             the final tree iterations. Each block's segment is
 *             COARSE_FACTOR × 2 × blockDim.x elements.
 *  Hardware:  GTX 1050, sm_61 (Pascal) — set device 1
 * =============================================================================
 *
 *  KERNEL CONCEPT
 *  ---------------------------------------------------------------------------
 *  - Each segment = COARSE_FACTOR * 2 * blockDim.x elements
 *  - Each thread independently accumulates COARSE_FACTOR pairs into a register
 *  - No __syncthreads() in the coarsening loop — threads work independently
 *  - Then stores result to shared memory and runs standard reduction tree
 *  - Reduces total steps: e.g. CF=2 gives 6 steps vs 8 for 2 uncoarsened blocks
 *  - 3 steps at full utilization (vs 2), 3 underutilized steps (vs 6)
 * =============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../common/cuda_utils.cuh"

#define COARSE_FACTOR 4

// =============================================================================
//  Fig 10.15 — Thread-coarsened reduction kernel
// =============================================================================
__global__ void coarsened_reduction_kernel(const float* input,
                                           float* output,
                                           int n) {
    __shared__ float input_s[1024];
    int t = threadIdx.x;

    // Segment for this block: COARSE_FACTOR × (2 × blockDim.x) elements
    int segment_size = COARSE_FACTOR * 2 * blockDim.x;
    int segment_start = blockIdx.x * segment_size;

    // Coarsening loop: each thread independently accumulates
    // COARSE_FACTOR pairs into a register
    float sum = 0.0f;
    for (int c = 0; c < COARSE_FACTOR; c++) {
        int i = segment_start + t + c * (2 * blockDim.x);
        if (i + blockDim.x < n) {
            sum += input[i] + input[i + blockDim.x];
        } else if (i < n) {
            sum += input[i];  // last odd element of this segment
        }
        // else: beyond input, skip
    }

    // Store partial sum to shared memory
    input_s[t] = sum;
    __syncthreads();

    // Standard reduction tree in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (t < stride) {
            input_s[t] += input_s[t + stride];
        }
        __syncthreads();
    }

    // Thread 0 atomically adds to global output
    if (t == 0) {
        atomicAdd(output, input_s[0]);
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

    // Input size
    const int BLOCK_SIZE = 256;
    const int ELEMS_PER_BLOCK = COARSE_FACTOR * 2 * BLOCK_SIZE;
    const int NUM_BLOCKS = 128;        // 128 blocks total
    const int N = NUM_BLOCKS * ELEMS_PER_BLOCK;

    printf("\n=== Fig 10.15: Thread-Coarsened Reduction (CF=%d) ===\n", COARSE_FACTOR);
    printf("Input size:    %d elements\n", N);
    printf("Block size:    %d threads\n", BLOCK_SIZE);
    printf("Blocks:        %d (vs %d without coarsening)\n",
           NUM_BLOCKS, NUM_BLOCKS * COARSE_FACTOR);
    printf("Elements/block: %d (vs %d without coarsening)\n",
           ELEMS_PER_BLOCK, 2 * BLOCK_SIZE);
    printf("Elements/thread: %d\n\n", 2 * COARSE_FACTOR);

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

    // Reset output to 0
    CHECK_CUDA(cudaMemset(d_output, 0, sizeof(float)));

    // Warm-up
    coarsened_reduction_kernel<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_input, d_output, N);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Reset output
    CHECK_CUDA(cudaMemset(d_output, 0, sizeof(float)));

    // Timed run
    gpu_timer timer;
    timer.start();
    coarsened_reduction_kernel<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_input, d_output, N);
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
           (float)(N * sizeof(float)) / (elapsed * 1e6f));
    printf("Validation:  %s\n", passed ? "PASSED" : "FAILED");
    printf("---------------------------------------------------------\n");

    // Cleanup
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));
    free(h_input);
    free(h_output);

    return passed ? 0 : 1;
}
