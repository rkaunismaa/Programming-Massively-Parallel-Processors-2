/*
 * Section 11.6 / Figs 11.9, 11.10 — Segmented Scan for Arbitrary-Length Inputs
 *
 * Three-kernel hierarchical scan algorithm for arbitrarily large inputs:
 *
 *   Kernel 1: Kogge-Stone scan per block + save last element to S[] (global)
 *   Kernel 2: Single-block Kogge-Stone scan on the S[] array
 *   Kernel 3: Add S[blockIdx.x-1] to every element of each block's output
 *
 * Each scan block (SECTION_SIZE elements) is processed independently by kernel 1.
 * The last element of each block is the sum of all elements in that block.
 * These are collected into array S, scanned, then used to add the contributions
 * from all preceding blocks to each block's elements.
 *
 * Hardware: GTX 1050 (sm_61 Pascal, device 1)
 * Compile:  nvcc -std=c++17 -arch=sm_61 -O2 -o ch11_segmented_scan ch11_segmented_scan.cu
 */

#include "../common/cuda_utils.cuh"
#include <cstdlib>
#include <ctime>

#define SECTION_SIZE 1024

// ---------------------------------------------------------------------------
// Kernel 1: Kogge-Stone scan per block + save last element to S[]
// ---------------------------------------------------------------------------
__global__ void segmented_scan_kernel1(const float *X, float *Y, float *S,
                                        unsigned int N, unsigned int num_blocks) {
    __shared__ float XY[SECTION_SIZE];

    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
        XY[threadIdx.x] = X[i];
    } else {
        XY[threadIdx.x] = 0.0f;
    }

    // Kogge-Stone scan
    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
        __syncthreads();
        float temp;
        if (threadIdx.x >= stride) {
            temp = XY[threadIdx.x] + XY[threadIdx.x - stride];
        }
        __syncthreads();
        if (threadIdx.x >= stride) {
            XY[threadIdx.x] = temp;
        }
    }

    __syncthreads();

    // Write scan result to Y
    if (i < N) {
        Y[i] = XY[threadIdx.x];
    }

    // Last thread in the block writes the last element (block sum) to S
    if (threadIdx.x == blockDim.x - 1 && blockIdx.x < num_blocks) {
        S[blockIdx.x] = XY[blockDim.x - 1];
    }
}

// ---------------------------------------------------------------------------
// Kernel 2: Single-block Kogge-Stone scan on S array
// ---------------------------------------------------------------------------
__global__ void segmented_scan_kernel2(float *S, unsigned int M) {
    // M = number of blocks (num_blocks)
    // Use as many threads as needed, up to SECTION_SIZE
    __shared__ float XY[SECTION_SIZE];

    unsigned int tid = threadIdx.x;

    if (tid < M) {
        XY[tid] = S[tid];
    } else {
        XY[tid] = 0.0f;
    }

    // Kogge-Stone scan (only need log2(M) iterations)
    for (unsigned int stride = 1; stride < blockDim.x && stride < M; stride *= 2) {
        __syncthreads();
        float temp;
        if (tid >= stride) {
            temp = XY[tid] + XY[tid - stride];
        }
        __syncthreads();
        if (tid >= stride) {
            XY[tid] = temp;
        }
    }

    __syncthreads();

    if (tid < M) {
        S[tid] = XY[tid];
    }
}

// ---------------------------------------------------------------------------
// Kernel 3: Add S[blockIdx.x-1] to every element of each block
//   Block 0's elements are already correct (no preceding blocks).
//   Each block's thread i reads X[i] + S[blockIdx.x-1].
// ---------------------------------------------------------------------------
__global__ void segmented_scan_kernel3(float *Y, const float *S,
                                        unsigned int N, unsigned int num_blocks) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    // Add the accumulated sum of all preceding blocks
    if (blockIdx.x > 0) {
        Y[i] += S[blockIdx.x - 1];
    }
}

// ---------------------------------------------------------------------------
// Host-side reference
// ---------------------------------------------------------------------------
void host_inclusive_scan(const float *X, float *Y, unsigned int N) {
    Y[0] = X[0];
    for (unsigned int i = 1; i < N; i++) {
        Y[i] = Y[i - 1] + X[i];
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main() {
    int dev = 1;
    cudaSetDevice(dev);
    print_device_info(dev);

    // Large input: 32 * SECTION_SIZE = 32768 elements, 32 blocks
    const unsigned int N = 32 * SECTION_SIZE;
    const unsigned int num_blocks = (N + SECTION_SIZE - 1) / SECTION_SIZE;

    float *h_X = new float[N];
    float *h_Y_gpu = new float[N];
    float *h_Y_ref = new float[N];

    srand(time(nullptr));
    for (unsigned int i = 0; i < N; i++) {
        h_X[i] = (float)(rand() % 100) / 10.0f;
    }

    host_inclusive_scan(h_X, h_Y_ref, N);

    // Device memory
    float *d_X, *d_Y, *d_S;
    CHECK_CUDA(cudaMalloc(&d_X, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_Y, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_S, num_blocks * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_X, h_X, N * sizeof(float), cudaMemcpyHostToDevice));

    dim3 grid(num_blocks, 1, 1);
    dim3 block(SECTION_SIZE, 1, 1);

    // Warm-up
    segmented_scan_kernel1<<<grid, block>>>(d_X, d_Y, d_S, N, num_blocks);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Restore d_X between warm-up and timed run (kernel1 doesn't modify d_X,
    // but d_S needs resetting)
    CHECK_CUDA(cudaMemset(d_S, 0, num_blocks * sizeof(float)));

    gpu_timer timer;
    timer.start();

    // Kernel 1: Per-block scan + write last elements to S
    segmented_scan_kernel1<<<grid, block>>>(d_X, d_Y, d_S, N, num_blocks);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Kernel 2: Single-block scan on S
    segmented_scan_kernel2<<<1, SECTION_SIZE>>>(d_S, num_blocks);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Kernel 3: Add S[blockIdx.x-1] to each block
    segmented_scan_kernel3<<<grid, block>>>(d_Y, d_S, N, num_blocks);
    CHECK_CUDA(cudaDeviceSynchronize());

    timer.stop();
    float ms = timer.elapsed_ms();

    CHECK_CUDA(cudaMemcpy(h_Y_gpu, d_Y, N * sizeof(float), cudaMemcpyDeviceToHost));

    bool passed = cpu_allclose(h_Y_ref, h_Y_gpu, N, 1e-4f);

    float bytes = (float)N * sizeof(float) * 2.0f;  // read X, write Y
    float bw = bytes / (ms * 1e-3f) / 1.0e9f;

    printf("\n");
    printf("---------------------------------------------------------\n");
    printf("Segmented Scan (3-kernel hierarchical, SECTION_SIZE=%d)\n", SECTION_SIZE);
    printf("---------------------------------------------------------\n");
    printf("  Input size    : %u elements\n", N);
    printf("  Blocks        : %u\n", num_blocks);
    printf("  Threads/block : %d\n", SECTION_SIZE);
    printf("  Total time    : %.3f ms (3 kernels)\n", ms);
    printf("  Effective BW  : %.2f GB/s\n", bw);
    printf("  Validation    : %s\n", passed ? "PASSED" : "FAILED");
    printf("---------------------------------------------------------\n");

    printf("\nFirst 12 results:\n");
    for (unsigned int i = 0; i < 12 && i < N; i++) {
        printf("  Y[%2u] = %.2f (expected %.2f)%s\n",
               i, h_Y_gpu[i], h_Y_ref[i],
               std::abs(h_Y_gpu[i] - h_Y_ref[i]) > 1e-4f ? " MISMATCH!" : "");
    }

    printf("\nCross-block boundary check:\n");
    for (unsigned int bi = 0; bi < 4 && bi <= num_blocks; bi++) {
        unsigned int idx = bi * SECTION_SIZE - 1;
        if (idx < N && bi > 0) {
            printf("  Y[%4u] (end of block %d) = %.2f (expected %.2f)%s\n",
                   idx, bi - 1, h_Y_gpu[idx], h_Y_ref[idx],
                   std::abs(h_Y_gpu[idx] - h_Y_ref[idx]) > 1e-4f ? " MISMATCH!" : "");
        }
        idx = bi * SECTION_SIZE;
        if (idx < N) {
            printf("  Y[%4u] (start of block %d) = %.2f (expected %.2f)%s\n",
                   idx, bi, h_Y_gpu[idx], h_Y_ref[idx],
                   std::abs(h_Y_gpu[idx] - h_Y_ref[idx]) > 1e-4f ? " MISMATCH!" : "");
        }
    }

    delete[] h_X;
    delete[] h_Y_gpu;
    delete[] h_Y_ref;
    CHECK_CUDA(cudaFree(d_X));
    CHECK_CUDA(cudaFree(d_Y));
    CHECK_CUDA(cudaFree(d_S));

    return passed ? 0 : 1;
}
