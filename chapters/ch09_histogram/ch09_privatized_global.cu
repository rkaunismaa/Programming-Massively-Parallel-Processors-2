/*
 * Chapter 9: Parallel Histogram
 * Privatized Histogram — Global Memory (Fig 9.9)
 *
 * Each thread block gets a private copy of the histogram in global memory.
 * Each thread atomically updates its block's private copy, then the
 * private copies are merged into the public copy (block 0's region).
 *
 * Benefits:
 *   - Contention is reduced from all-threads to per-block
 *   - Private copies are L2-cached, reducing latency
 *
 * Trade-off: merge phase adds overhead proportional to number of blocks.
 *
 * Compute: GTX 1050, sm_61 (device 1)
 */

#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <cstring>
#include "../common/cuda_utils.cuh"

#define NUM_BINS 7
#define ALPHABET_SIZE 26
#define LETTERS_PER_BIN 4

/*
 * Privatized histogram kernel — global memory (Fig 9.9).
 *
 * Histogram array layout: gridDim.x blocks × NUM_BINS bins
 * Each block writes to its own private region: histo[blockIdx.x * NUM_BINS + bin]
 *
 * After computation, block 0's region becomes the public merged histogram.
 *
 * Parameters:
 *   data   - input character array
 *   histo  - private histogram array (size: gridDim.x * NUM_BINS)
 *   length - number of input elements
 */
__global__ void privatized_global_histogram_kernel(
    const char* __restrict__ data,
    int* __restrict__ histo,
    int length)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < length) {
        int pos = data[tid] - 'a';
        if (pos >= 0 && pos < ALPHABET_SIZE) {
            int bin = pos / LETTERS_PER_BIN;
            // Offset to this block's private copy
            int offset = blockIdx.x * NUM_BINS + bin;
            atomicAdd(&histo[offset], 1);
        }
    }

    __syncthreads();

    // Merge: thread 0 commits this block's private copy to block 0's public copy
    if (threadIdx.x == 0 && blockIdx.x > 0) {
        for (int b = 0; b < NUM_BINS; ++b) {
            int val = histo[blockIdx.x * NUM_BINS + b];
            if (val > 0) {
                atomicAdd(&histo[b], val);
            }
        }
    }
}

/* -------------------------------------------------------------------------- */
/*  CPU reference                                                             */
/* -------------------------------------------------------------------------- */
void cpu_histogram(const char* data, int* histo, int length)
{
    for (int i = 0; i < NUM_BINS; ++i) histo[i] = 0;
    for (int i = 0; i < length; ++i) {
        int pos = data[i] - 'a';
        if (pos >= 0 && pos < ALPHABET_SIZE) {
            histo[pos / LETTERS_PER_BIN]++;
        }
    }
}

/* -------------------------------------------------------------------------- */
/*  Utility: generate biased text                                             */
/* -------------------------------------------------------------------------- */
void generate_text(char* data, int length)
{
    for (int i = 0; i < length; ++i) {
        int r = rand() % 100;
        if (r < 60)
            data[i] = 'i' + (rand() % 10);
        else if (r < 85)
            data[i] = 'a' + (rand() % 8);
        else
            data[i] = 's' + (rand() % 8);
    }
}

/* -------------------------------------------------------------------------- */
/*  Main                                                                      */
/* -------------------------------------------------------------------------- */
int main()
{
    cudaSetDevice(1);
    CHECK_CUDA(cudaGetLastError());

    const int N = 1 << 24;  // 16M characters

    std::cout << "=================================================\n";
    std::cout << "Chapter 9: Privatized Histogram (Fig 9.9)\n";
    std::cout << "=================================================\n";
    std::cout << "Input: " << N << " characters (" << (N / (1024*1024)) << " MB)\n";
    std::cout << "Bins:  " << NUM_BINS << "\n";
    std::cout << "Device: GTX 1050 (sm_61)\n\n";

    // Host
    char* h_data = new char[N];
    int* h_histo = new int[NUM_BINS];
    int* h_ref   = new int[NUM_BINS];

    srand(42);
    generate_text(h_data, N);
    cpu_histogram(h_data, h_ref, N);

    // Device
    char* d_data = nullptr;
    int* d_histo = nullptr;

    int block_size = 256;
    int grid_size = (N + block_size - 1) / block_size;

    // Allocate space for private copies: [grid_size * NUM_BINS] ints
    int private_histo_size = grid_size * NUM_BINS;

    CHECK_CUDA(cudaMalloc(&d_data,  N * sizeof(char)));
    CHECK_CUDA(cudaMalloc(&d_histo, private_histo_size * sizeof(int)));
    CHECK_CUDA(cudaMemcpy(d_data, h_data, N * sizeof(char), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_histo, 0, private_histo_size * sizeof(int)));

    std::cout << "Block size: " << block_size << "\n";
    std::cout << "Grid size:  " << grid_size << "\n";
    std::cout << "Private histo entries: " << private_histo_size
              << " (" << (private_histo_size * sizeof(int)) << " bytes)\n\n";

    // Warm-up
    privatized_global_histogram_kernel<<<grid_size, block_size>>>(d_data, d_histo, N);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Reset for timed run
    CHECK_CUDA(cudaMemset(d_histo, 0, private_histo_size * sizeof(int)));

    gpu_timer timer;
    timer.start();
    privatized_global_histogram_kernel<<<grid_size, block_size>>>(d_data, d_histo, N);
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();

    float kernel_time = timer.elapsed_ms();

    // Read back the public copy (first NUM_BINS elements)
    CHECK_CUDA(cudaMemcpy(h_histo, d_histo, NUM_BINS * sizeof(int), cudaMemcpyDeviceToHost));

    // Print
    char bin_names[][10] = {"a-d", "e-h", "i-l", "m-p", "q-t", "u-x", "y-z"};
    std::cout << "Histogram:\n";
    for (int i = 0; i < NUM_BINS; ++i) {
        std::cout << "  " << bin_names[i] << ": " << h_histo[i]
                  << " (ref: " << h_ref[i] << ")"
                  << (h_histo[i] == h_ref[i] ? "" : " MISMATCH") << "\n";
    }

    bool valid = true;
    for (int i = 0; i < NUM_BINS; ++i) {
        if (h_histo[i] != h_ref[i]) { valid = false; break; }
    }

    float atomics_per_sec = N / (kernel_time * 1e-3f) / 1e6f;

    std::cout << "\nResults:\n";
    std::cout << "  Kernel time:     " << std::fixed << std::setprecision(2) << kernel_time << " ms\n";
    std::cout << "  Atomics/sec:     " << std::fixed << std::setprecision(1) << atomics_per_sec << " M\n";
    std::cout << "  Validation:      " << (valid ? "PASSED" : "FAILED") << "\n\n";

    CHECK_CUDA(cudaFree(d_data));
    CHECK_CUDA(cudaFree(d_histo));
    delete[] h_data;
    delete[] h_histo;
    delete[] h_ref;

    return valid ? 0 : 1;
}
