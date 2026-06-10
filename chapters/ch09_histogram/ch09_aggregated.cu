/*
 * Chapter 9: Parallel Histogram
 * Aggregated Histogram (Fig 9.15)
 *
 * When many consecutive characters map to the same histogram bin
 * (e.g., large patches of identical letters), each thread aggregates
 * consecutive same-bin updates into a single atomic operation.
 *
 * Algorithm per thread:
 *   1. Read next input element
 *   2. If same bin as previous → increment local accumulator (streak)
 *   3. If different bin → flush accumulator via atomicAdd, start new streak
 *   4. At end → flush remaining accumulator
 *
 * Builds on the interleaved-partitioning coarsened kernel.
 *
 * Benefit: reduces atomic operations for biased/streaky data
 * Trade-off: more code, control divergence when some threads flush
 *            while others continue their streak
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
#define COARSE_FACTOR 4

/*
 * Aggregated histogram kernel (Fig 9.15).
 *
 * Each thread accumulates consecutive same-bin updates before performing
 * a single atomicAdd. This reduces contention for streaky data.
 *
 * Parameters:
 *   data   - input character array
 *   histo  - public histogram (NUM_BINS elements, device)
 *   length - number of input elements
 */
__global__ void aggregated_histogram_kernel(
    const char* __restrict__ data,
    int* __restrict__ histo,
    int length)
{
    __shared__ int histo_s[NUM_BINS];

    // Initialize shared memory private histogram
    if (threadIdx.x < NUM_BINS) {
        histo_s[threadIdx.x] = 0;
    }
    __syncthreads();

    // Per-thread aggregation state
    int accumulator = 0;      // streak count being aggregated
    int prevBinIdx = -1;      // index of the bin being aggregated (-1 = none)

    // Interleaved processing
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = 0; i < COARSE_FACTOR; ++i) {
        int idx = tid + i * stride;
        if (idx < length) {
            int pos = data[idx] - 'a';
            if (pos >= 0 && pos < ALPHABET_SIZE) {
                int bin = pos / LETTERS_PER_BIN;

                if (bin == prevBinIdx) {
                    // Same bin — extend the streak
                    accumulator++;
                } else {
                    // Different bin — flush the old streak
                    if (prevBinIdx >= 0 && accumulator > 0) {
                        atomicAdd(&histo_s[prevBinIdx], accumulator);
                    }
                    // Start new streak
                    prevBinIdx = bin;
                    accumulator = 1;
                }
            } else {
                // Non-alphabet character — flush any pending streak
                if (prevBinIdx >= 0 && accumulator > 0) {
                    atomicAdd(&histo_s[prevBinIdx], accumulator);
                    prevBinIdx = -1;
                    accumulator = 0;
                }
            }
        }
    }

    // Flush any remaining streak
    if (prevBinIdx >= 0 && accumulator > 0) {
        atomicAdd(&histo_s[prevBinIdx], accumulator);
    }

    __syncthreads();

    // Merge to public histogram
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

    std::cout << "================================================\n";
    std::cout << "Chapter 9: Aggregated Histogram (Fig 9.15)\n";
    std::cout << "================================================\n";
    std::cout << "Input: " << N << " characters (" << (N / (1024*1024)) << " MB)\n";
    std::cout << "Coarse factor: " << COARSE_FACTOR << "\n";
    std::cout << "Device: GTX 1050 (sm_61)\n\n";

    char* h_data = new char[N];
    int* h_histo = new int[NUM_BINS];
    int* h_ref   = new int[NUM_BINS];

    srand(42);
    generate_text(h_data, N);
    cpu_histogram(h_data, h_ref, N);

    char* d_data = nullptr;
    int* d_histo = nullptr;

    int block_size = 256;
    int grid_size = (N + block_size * COARSE_FACTOR - 1) / (block_size * COARSE_FACTOR);

    CHECK_CUDA(cudaMalloc(&d_data,  N * sizeof(char)));
    CHECK_CUDA(cudaMalloc(&d_histo, NUM_BINS * sizeof(int)));
    CHECK_CUDA(cudaMemcpy(d_data, h_data, N * sizeof(char), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_histo, 0, NUM_BINS * sizeof(int)));

    int total_threads = block_size * grid_size;
    std::cout << "Block size: " << block_size << "\n";
    std::cout << "Grid size:  " << grid_size << "\n";
    std::cout << "Stride:     " << total_threads << "\n\n";

    // Warm-up
    aggregated_histogram_kernel<<<grid_size, block_size>>>(d_data, d_histo, N);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemset(d_histo, 0, NUM_BINS * sizeof(int)));

    gpu_timer timer;
    timer.start();
    aggregated_histogram_kernel<<<grid_size, block_size>>>(d_data, d_histo, N);
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
    std::cout << "  Kernel time:       " << std::fixed << std::setprecision(2) << kernel_time << " ms\n";
    std::cout << "  Atomics/sec:       " << std::fixed << std::setprecision(1) << atomics_per_sec << " M\n";
    std::cout << "  Validation:        " << (valid ? "PASSED" : "FAILED") << "\n\n";

    CHECK_CUDA(cudaFree(d_data));
    CHECK_CUDA(cudaFree(d_histo));
    delete[] h_data;
    delete[] h_histo;
    delete[] h_ref;

    return valid ? 0 : 1;
}
