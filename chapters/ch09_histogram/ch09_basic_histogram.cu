/*
 * Chapter 9: Parallel Histogram
 * Basic Histogram with Atomic Operations — Fig 9.6
 *
 * Each thread reads one input element and atomically increments
 * the appropriate histogram bin in global memory.
 *
 * Text input: a random-generated string of lowercase letters.
 * Histogram: 7 bins (7 intervals of 4 letters each: a-d, e-h, ..., y-z*)
 *   * The 7th bin (y-z) is only 2 letters wide.
 *
 * Key concept: atomicAdd ensures correctness but serializes
 * concurrent updates to the same bin, creating a bottleneck.
 *
 * Compute: GTX 1050, sm_61 (device 1)
 */

#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <cstring>
#include <cfloat>
#include "../common/cuda_utils.cuh"

#define NUM_BINS 7
#define ALPHABET_SIZE 26
#define LETTERS_PER_BIN 4

/*
 * Basic histogram kernel (Fig 9.6).
 *
 * Each thread processes one input character using atomicAdd.
 *
 * Parameters:
 *   data  - input character array (length elements)
 *   histo - output histogram array (NUM_BINS elements, device)
 *   length - number of input elements
 */
__global__ void basic_histogram_kernel(
    const char* __restrict__ data,
    int* __restrict__ histo,
    int length)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < length) {
        char c = data[tid];
        int alphabet_position = c - 'a';
        if (alphabet_position >= 0 && alphabet_position < ALPHABET_SIZE) {
            int bin = alphabet_position / LETTERS_PER_BIN;
            atomicAdd(&histo[bin], 1);
        }
    }
}

/* -------------------------------------------------------------------------- */
/*  CPU reference: sequential histogram                                       */
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
/*  Utility: generate random lowercase text                                   */
/* -------------------------------------------------------------------------- */
void generate_text(char* data, int length)
{
    for (int i = 0; i < length; ++i) {
        // Bias toward middle bins to create contention (like real text)
        int r = rand() % 100;
        if (r < 60)  // 60% chance: middle bins (i-r: 8..17)
            data[i] = 'i' + (rand() % 10);
        else if (r < 85)  // 25%: early letters (a-h: 0..7)
            data[i] = 'a' + (rand() % 8);
        else  // 15%: late letters (s-z: 18..25)
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

    // Input size
    const int N = 1 << 24;  // 16M characters
    const int total_bins = NUM_BINS;  // only 7 bins

    std::cout << "=============================================\n";
    std::cout << "Chapter 9: Basic Histogram (Fig 9.6)\n";
    std::cout << "=============================================\n";
    std::cout << "Input: " << N << " characters (" << (N / (1024*1024)) << " MB)\n";
    std::cout << "Bins:  " << NUM_BINS << " (4 letters per bin, last bin 2 letters)\n";
    std::cout << "Device: GTX 1050 (sm_61)\n\n";

    // Host memory
    char* h_data = new char[N];
    int* h_histo = new int[NUM_BINS];
    int* h_ref   = new int[NUM_BINS];

    // Generate biased text
    srand(42);
    generate_text(h_data, N);

    // CPU reference
    cpu_histogram(h_data, h_ref, N);

    // Device memory
    char* d_data = nullptr;
    int* d_histo = nullptr;
    CHECK_CUDA(cudaMalloc(&d_data,  N * sizeof(char)));
    CHECK_CUDA(cudaMalloc(&d_histo, NUM_BINS * sizeof(int)));
    CHECK_CUDA(cudaMemcpy(d_data, h_data, N * sizeof(char), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_histo, 0, NUM_BINS * sizeof(int)));

    // Launch configuration
    int block_size = 256;
    int grid_size = (N + block_size - 1) / block_size;

    std::cout << "Block size: " << block_size << "\n";
    std::cout << "Grid size:  " << grid_size << "\n\n";

    // Warm-up run
    basic_histogram_kernel<<<grid_size, block_size>>>(d_data, d_histo, N);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Reset histogram for timed run
    CHECK_CUDA(cudaMemset(d_histo, 0, NUM_BINS * sizeof(int)));

    // Timed run
    gpu_timer timer;
    timer.start();
    basic_histogram_kernel<<<grid_size, block_size>>>(d_data, d_histo, N);
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();

    float kernel_time = timer.elapsed_ms();

    // Copy result back
    CHECK_CUDA(cudaMemcpy(h_histo, d_histo, NUM_BINS * sizeof(int), cudaMemcpyDeviceToHost));

    // Print histogram
    std::cout << "Histogram:\n";
    char bin_names[NUM_BINS][10] = {"a-d", "e-h", "i-l", "m-p", "q-t", "u-x", "y-z"};
    for (int i = 0; i < NUM_BINS; ++i) {
        std::cout << "  " << bin_names[i] << ": " << h_histo[i]
                  << " (ref: " << h_ref[i] << ")"
                  << (h_histo[i] == h_ref[i] ? "" : " MISMATCH")
                  << "\n";
    }

    // Validate
    bool valid = true;
    for (int i = 0; i < NUM_BINS; ++i) {
        if (h_histo[i] != h_ref[i]) { valid = false; break; }
    }

    // Compute throughput
    float atomics_per_sec = N / (kernel_time * 1e-3f) / 1e6f;

    std::cout << "\nResults:\n";
    std::cout << "  Kernel time:     " << std::fixed << std::setprecision(2) << kernel_time << " ms\n";
    std::cout << "  Atomics/sec:     " << std::fixed << std::setprecision(1) << atomics_per_sec << " M\n";
    std::cout << "  Validation:      " << (valid ? "PASSED" : "FAILED") << "\n\n";

    // Cleanup
    CHECK_CUDA(cudaFree(d_data));
    CHECK_CUDA(cudaFree(d_histo));
    delete[] h_data;
    delete[] h_histo;
    delete[] h_ref;

    return valid ? 0 : 1;
}
