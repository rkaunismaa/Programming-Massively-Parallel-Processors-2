/*
 * Chapter 9: Parallel Histogram
 * Privatized Histogram — Shared Memory (Fig 9.10)
 *
 * Each thread block gets a private copy of the histogram in shared memory.
 * Shared memory offers much lower latency for atomic operations than
 * global memory, dramatically improving throughput.
 *
 * Steps:
 *   1. All threads initialize private histogram in shared mem (all bins = 0)
 *   2. __syncthreads()
 *   3. Each thread processes its input element, calling atomicAdd on shared mem
 *   4. __syncthreads()
 *   5. Each thread commits one or more private bins to the public histogram
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
 * Privatized histogram kernel — shared memory (Fig 9.10).
 *
 * Private histogram stored in histo_s (shared).
 * Merge: each thread commits bin = threadIdx.x % NUM_BINS iteratively.
 *
 * Parameters:
 *   data   - input character array
 *   histo  - public histogram (NUM_BINS elements, device)
 *   length - number of input elements
 */
__global__ void privatized_shared_histogram_kernel(
    const char* __restrict__ data,
    int* __restrict__ histo,
    int length)
{
    // Private histogram in shared memory
    __shared__ int histo_s[NUM_BINS];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Initialize shared memory private histogram
    if (threadIdx.x < NUM_BINS) {
        histo_s[threadIdx.x] = 0;
    }
    __syncthreads();

    // Process input element — atomicAdd on SHARED memory (fast)
    if (tid < length) {
        int pos = data[tid] - 'a';
        if (pos >= 0 && pos < ALPHABET_SIZE) {
            int bin = pos / LETTERS_PER_BIN;
            atomicAdd(&histo_s[bin], 1);
        }
    }

    __syncthreads();

    // Merge: threads cooperatively commit private bins to public histogram
    // Each thread handles bins spaced by blockDim.x
    for (int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x) {
        int val = histo_s[bin];
        if (val > 0) {
            atomicAdd(&histo[bin], val);
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

    const int N = 1 << 24;

    std::cout << "=================================================\n";
    std::cout << "Chapter 9: Privatized Histogram (Fig 9.10)\n";
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

    CHECK_CUDA(cudaMalloc(&d_data,  N * sizeof(char)));
    CHECK_CUDA(cudaMalloc(&d_histo, NUM_BINS * sizeof(int)));
    CHECK_CUDA(cudaMemcpy(d_data, h_data, N * sizeof(char), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_histo, 0, NUM_BINS * sizeof(int)));

    std::cout << "Block size: " << block_size << "\n";
    std::cout << "Grid size:  " << grid_size << "\n";
    std::cout << "Shared mem: " << (NUM_BINS * sizeof(int)) << " B per block\n\n";

    // Warm-up
    privatized_shared_histogram_kernel<<<grid_size, block_size>>>(d_data, d_histo, N);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Reset
    CHECK_CUDA(cudaMemset(d_histo, 0, NUM_BINS * sizeof(int)));

    gpu_timer timer;
    timer.start();
    privatized_shared_histogram_kernel<<<grid_size, block_size>>>(d_data, d_histo, N);
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();

    float kernel_time = timer.elapsed_ms();

    CHECK_CUDA(cudaMemcpy(h_histo, d_histo, NUM_BINS * sizeof(int), cudaMemcpyDeviceToHost));

    char bin_names[][10] = {"a-d", "e-h", "i-l", "m-p", "q-t", "u-x", "y-z"};
    std::cout << "Histogram:\n";
    for (int i = 0; i < NUM_BINS; ++i) {
        std::cout << "  " << bin_names[i] << ": " << h_histo[i]
                  << " (ref: " << h_ref[i] << ")"
                  << (h_histo[i] == h_ref[i] ? "" : " MISMATCH") << "\n";
    }

    bool valid = true;
    for (int i = 0; i < NUM_BINS; ++i)
        if (h_histo[i] != h_ref[i]) { valid = false; break; }

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
