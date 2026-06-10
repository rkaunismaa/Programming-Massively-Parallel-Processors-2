/*
 * =============================================================================
 *  Chapter 10: Parallel Reduction — Fig 10.13: Multiblock Segmented Reduction
 * =============================================================================
 *  Book:      Programming Massively Parallel Processors (Kirk, Hwu & El Hajj,
 *             4th Edition, Morgan Kaufmann 2023)
 *  Figure:    10.13 — Segmented multiblock sum reduction using atomic operations
 *  Purpose:   Extends the shared-memory reduction to arbitrary input sizes.
 *             Partitions input into segments of 2*blockDim.x elements per block.
 *             Each block independently computes the sum of its segment in shared
 *             memory, then the first thread of each block atomically adds its
 *             partial sum to the global output.
 *  Hardware:  GTX 1050, sm_61 (Pascal) — set device 1
 * =============================================================================
 *
 *  KERNEL CONCEPT
 *  ---------------------------------------------------------------------------
 *  - Partition input into segments of 2*blockDim.x elements
 *  - Each segment is processed by one block using shared-memory reduction
 *  - Atomic add on output ensures correctness across blocks
 *  - Works for arbitrarily large inputs (millions/billions of elements)
 *  - Segment size must be a power of 2 (but input N can be any value)
 * =============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../common/cuda_utils.cuh"

// =============================================================================
//  Fig 10.13 — Multiblock segmented reduction kernel
// =============================================================================
__global__ void multiblock_reduction_kernel(const float* input,
                                            float* output,
                                            int n) {
    __shared__ float input_s[1024];
    int t = threadIdx.x;

    // Segment start for this block
    int segment_size = 2 * blockDim.x;
    int segment_start = blockIdx.x * segment_size;
    int i = segment_start + t;  // global index for this thread

    // Load and add pair (guard against partial segments)
    if (i + blockDim.x < n) {
        input_s[t] = input[i] + input[i + blockDim.x];
    } else if (i < n) {
        input_s[t] = input[i];  // last odd element
    } else {
        input_s[t] = 0.0f;  // beyond input
    }
    __syncthreads();

    // Remaining iterations in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (t < stride) {
            input_s[t] += input_s[t + stride];
        }
        __syncthreads();
    }

    // Thread 0 atomically adds partial sum to global output
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

    // Input size — large, arbitrary (multiple segments)
    const int BLOCK_SIZE = 256;
    const int SEGMENT_SIZE = 2 * BLOCK_SIZE;
    const int NUM_BLOCKS = 256;       // 256 blocks × 512 elements = 131,072
    const int N = NUM_BLOCKS * SEGMENT_SIZE;

    printf("\n=== Fig 10.13: Multiblock Segmented Reduction (Atomic Add) ===\n");
    printf("Input size:  %d elements\n", N);
    printf("Block size:  %d threads\n", BLOCK_SIZE);
    printf("Blocks:      %d\n", NUM_BLOCKS);
    printf("Segment:     %d elements per block\n", SEGMENT_SIZE);
    printf("Iterations:  %d per block\n\n", (int)log2((float)SEGMENT_SIZE));

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
    multiblock_reduction_kernel<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_input, d_output, N);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Reset output
    CHECK_CUDA(cudaMemset(d_output, 0, sizeof(float)));

    // Timed run
    gpu_timer timer;
    timer.start();
    multiblock_reduction_kernel<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_input, d_output, N);
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
