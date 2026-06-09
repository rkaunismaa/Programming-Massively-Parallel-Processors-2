/*
 * =============================================================================
 *  Chapter 6: Performance Considerations — Memory Coalescing
 * =============================================================================
 *  Book:      Programming Massively Parallel Processors (Kirk, Hwu & El Hajj,
 *             4th Edition, Morgan Kaufmann 2023)
 *  Section:   6.1 — Memory Coalescing
 *  Purpose:   Benchmark coalesced vs. uncoalesced global-memory access
 *             patterns to quantify the bandwidth penalty of strided access.
 *  Hardware:  GTX 1050, sm_61 (Pascal) — all code targets this GPU
 * =============================================================================
 *
 *  MEMORY COALESCING CONCEPT
 *  ---------------------------------------------------------------------------
 *  On Pascal (and all modern GPUs), the memory controller can combine the
 *  individual 4-byte loads/stores of threads within a warp into a small set
 *  of contiguous DRAM transactions — provided the addresses form a contiguous
 *  region.  This is called *coalescing*.
 *
 *  COALESCED ACCESS (ideal):
 *    Thread i in a warp reads   input[i]
 *    Thread i+1 in a warp reads input[i+1]
 *    ...
 *    The 32 threads of a warp touch 32 consecutive floats (128 bytes).
 *    The memory controller issues ONE 128-byte transaction per warp.
 *    Bandwidth utilisation approaches 100%.
 *
 *  UNCOALESCED ACCESS (strided, stride=32 floats = 128 bytes):
 *    Thread i in a warp reads   input[i * 32]
 *    Thread i+1 in a warp reads input[(i+1) * 32]
 *    ...
 *    The 32 threads of a warp touch 32 addresses scattered across 32
 *    different 128-byte segments.  The memory controller must issue 32
 *    separate 128-byte transactions to serve ONE warp — a 32x increase
 *    in memory transactions for the same number of useful data words.
 *    Bandwidth utilisation drops to ~3%.
 *
 *  WHY COALESCED ACCESS IS FASTER:
 *    - DRAM has high latency (~200-500 cycles per access).  Coalescing
 *      amortises that latency across all 32 threads in a warp.
 *    - Each coalesced transaction transfers 128 bytes with ONE DRAM
 *      request.  Uncoalesced access needs 32 DRAM requests to transfer
 *      the same 128 bytes of useful data (the rest is wasted bandwidth).
 *    - The memory bus is a shared resource; fewer transactions mean less
 *      contention and lower overall latency for the entire grid.
 * =============================================================================
 */

#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstdlib>
#include <numeric>
#include "../common/cuda_utils.cuh"

/* -------------------------------------------------------------------------- */
/*  Kernel 1 — Coalesced access pattern                                       */
/*  Threads in a warp access consecutive memory locations.                     */
/*  The memory controller merges all 32 warp loads into a single transaction.  */
/* -------------------------------------------------------------------------- */
__global__ void coalesced_access(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[idx] * 2.0f;
    }
}

/* -------------------------------------------------------------------------- */
/*  Kernel 2 — Uncoalesced access pattern (stride = 32 floats = 128 bytes)     */
/*  Threads in a warp access memory locations 128 bytes apart.                 */
/*  Each thread's load falls in a different 128-byte segment, forcing the      */
/*  memory controller to issue 32 separate DRAM transactions per warp.         */
/* -------------------------------------------------------------------------- */
__global__ void uncoalesced_access(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = 32;
    if (idx < n) {
        output[idx * stride] = input[idx * stride] * 2.0f;
    }
}

/* -------------------------------------------------------------------------- */
/*  Main — benchmark and compare                                              */
/* -------------------------------------------------------------------------- */
int main() {
    const int DEVICE = 1;          // GTX 1050
    const int N = 1 << 20;         // 1M elements
    const int STRIDE = 32;         // stride for uncoalesced access
    const int BLOCK_SIZE = 256;    // threads per block
    const int NUM_ITERATIONS = 50; // repeat for stable timing

    // The uncoalesced kernel accesses input/output at indices 0, stride,
    // 2*stride, ..., (N-1)*stride.  So the arrays must be N*STRIDE elements
    // long to avoid out-of-bounds access.
    const int LARGE_N = N * STRIDE;  // 32M elements = 128 MB

    cudaSetDevice(DEVICE);
    print_device_info(DEVICE);

    /* ------------------------------------------------------------------ */
    /*  Allocate host memory                                               */
    /* ------------------------------------------------------------------ */
    float* h_input = new float[LARGE_N];
    float* h_expected_coalesced = new float[N];
    float* h_expected_uncoalesced = new float[LARGE_N];

    // Initialise input with sequential values: 0.0, 1.0, 2.0, ...
    std::iota(h_input, h_input + LARGE_N, 0.0f);

    // Compute expected results on CPU for verification
    // Coalesced: output[i] = input[i] * 2.0f  for i in [0, N)
    for (int i = 0; i < N; i++) {
        h_expected_coalesced[i] = h_input[i] * 2.0f;
    }
    // Uncoalesced: output[i*stride] = input[i*stride] * 2.0f
    // Only the strided positions are written; the rest stay zero.
    std::memset(h_expected_uncoalesced, 0, LARGE_N * sizeof(float));
    for (int i = 0; i < N; i++) {
        h_expected_uncoalesced[i * STRIDE] = h_input[i * STRIDE] * 2.0f;
    }

    /* ------------------------------------------------------------------ */
    /*  Allocate device memory                                             */
    /* ------------------------------------------------------------------ */
    float* d_input = nullptr;
    float* d_output_coalesced = nullptr;
    float* d_output_uncoalesced = nullptr;

    size_t large_input_bytes = LARGE_N * sizeof(float);
    size_t coalesced_out_bytes = N * sizeof(float);
    size_t large_out_bytes = LARGE_N * sizeof(float);

    CHECK_CUDA(cudaMalloc(&d_input, large_input_bytes));
    CHECK_CUDA(cudaMalloc(&d_output_coalesced, coalesced_out_bytes));
    CHECK_CUDA(cudaMalloc(&d_output_uncoalesced, large_out_bytes));

    // Copy input to device
    CHECK_CUDA(cudaMemcpy(d_input, h_input, large_input_bytes,
                          cudaMemcpyHostToDevice));

    /* ------------------------------------------------------------------ */
    /*  Launch configuration                                              */
    /* ------------------------------------------------------------------ */
    int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    std::cout << "  Grid: " << num_blocks << " blocks x "
              << BLOCK_SIZE << " threads = "
              << num_blocks * BLOCK_SIZE << " total threads\n";
    std::cout << "  Input array: " << LARGE_N << " elements ("
              << large_input_bytes / (1024 * 1024) << " MB)\n";
    std::cout << "  Coalesced output: " << N << " elements ("
              << coalesced_out_bytes / (1024 * 1024) << " MB)\n";
    std::cout << "  Uncoalesced output: " << LARGE_N << " elements ("
              << large_out_bytes / (1024 * 1024) << " MB)\n\n";

    /* ------------------------------------------------------------------ */
    /*  Warm-up run (avoid launch overhead in timing)                      */
    /* ------------------------------------------------------------------ */
    coalesced_access<<<num_blocks, BLOCK_SIZE>>>(d_input, d_output_coalesced, N);
    uncoalesced_access<<<num_blocks, BLOCK_SIZE>>>(d_input, d_output_uncoalesced, N);
    CHECK_CUDA(cudaDeviceSynchronize());

    /* ------------------------------------------------------------------ */
    /*  Benchmark — coalesced access                                       */
    /* ------------------------------------------------------------------ */
    gpu_timer timer_coalesced;
    timer_coalesced.start();
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        coalesced_access<<<num_blocks, BLOCK_SIZE>>>(d_input, d_output_coalesced, N);
    }
    timer_coalesced.stop();
    float coalesced_total_ms = timer_coalesced.elapsed_ms();
    float coalesced_avg_ms = coalesced_total_ms / NUM_ITERATIONS;

    /* ------------------------------------------------------------------ */
    /*  Benchmark — uncoalesced access                                     */
    /* ------------------------------------------------------------------ */
    gpu_timer timer_uncoalesced;
    timer_uncoalesced.start();
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        uncoalesced_access<<<num_blocks, BLOCK_SIZE>>>(d_input, d_output_uncoalesced, N);
    }
    timer_uncoalesced.stop();
    float uncoalesced_total_ms = timer_uncoalesced.elapsed_ms();
    float uncoalesced_avg_ms = uncoalesced_total_ms / NUM_ITERATIONS;

    /* ------------------------------------------------------------------ */
    /*  Verify results                                                     */
    /* ------------------------------------------------------------------ */
    float* h_output_coalesced = new float[N];
    float* h_output_uncoalesced = new float[LARGE_N];

    CHECK_CUDA(cudaMemcpy(h_output_coalesced, d_output_coalesced,
                          coalesced_out_bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_output_uncoalesced, d_output_uncoalesced,
                          large_out_bytes, cudaMemcpyDeviceToHost));

    bool coalesced_ok = cpu_allclose(h_expected_coalesced, h_output_coalesced, N);
    bool uncoalesced_ok = cpu_allclose(h_expected_uncoalesced, h_output_uncoalesced,
                                       LARGE_N);

    /* ------------------------------------------------------------------ */
    /*  Compute effective bandwidth                                        */
    /* ------------------------------------------------------------------ */
    // Each kernel processes N elements: reads N floats + writes N floats
    // = 2 * N * 4 bytes of useful data transfer per run.
    // The uncoalesced kernel transfers the same useful data but wastes
    // bandwidth because the hardware must service STRIDE times more
    // DRAM transactions (each transaction only carries 1 useful word).
    double useful_bytes_per_run = 2.0 * N * sizeof(float);
    double coalesced_bandwidth = (useful_bytes_per_run / (1024.0 * 1024.0)) /
                                 (coalesced_avg_ms / 1000.0);  // MB/s
    double uncoalesced_bandwidth = (useful_bytes_per_run / (1024.0 * 1024.0)) /
                                   (uncoalesced_avg_ms / 1000.0);  // MB/s

    /* ------------------------------------------------------------------ */
    /*  Print results                                                      */
    /* ------------------------------------------------------------------ */
    std::cout << "\n============================================================\n";
    std::cout << "  MEMORY COALESCING BENCHMARK RESULTS\n";
    std::cout << "============================================================\n";
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "  Iterations per test : " << NUM_ITERATIONS << "\n\n";

    std::cout << "  Coalesced access:\n";
    std::cout << "    Average time      : " << coalesced_avg_ms << " ms\n";
    std::cout << "    Verification      : "
              << (coalesced_ok ? "PASS" : "FAIL") << "\n";
    std::cout << "    Effective BW      : " << coalesced_bandwidth << " MB/s\n\n";

    std::cout << "  Uncoalesced access (stride=" << STRIDE << "):\n";
    std::cout << "    Average time      : " << uncoalesced_avg_ms << " ms\n";
    std::cout << "    Verification      : "
              << (uncoalesced_ok ? "PASS" : "FAIL") << "\n";
    std::cout << "    Effective BW      : " << uncoalesced_bandwidth << " MB/s\n\n";

    double speedup = uncoalesced_avg_ms / coalesced_avg_ms;
    std::cout << "  Speedup (uncoalesced/coalesced): " << speedup << "x\n";
    std::cout << "============================================================\n\n";

    // Explanation printed to help the reader understand the result
    std::cout << "  WHY COALESCED IS FASTER:\n";
    std::cout << "  - Coalesced:  32 threads in a warp touch 128 consecutive bytes.\n";
    std::cout << "    The memory controller issues ONE 128-byte DRAM transaction.\n";
    std::cout << "    Bandwidth utilisation ~100%.\n";
    std::cout << "  - Uncoalesced: 32 threads in a warp touch 32 different 128-byte\n";
    std::cout << "    segments (stride=" << STRIDE << " floats).  The controller must issue 32\n";
    std::cout << "    separate DRAM transactions.  Bandwidth utilisation ~"
              << (100.0 / STRIDE) << "%.\n";
    std::cout << "  - Theoretical speedup ~" << STRIDE << "x; actual speedup is typically\n";
    std::cout << "    lower due to warp-level scheduling and memory-level parallelism.\n\n";

    /* ------------------------------------------------------------------ */
    /*  Cleanup                                                            */
    /* ------------------------------------------------------------------ */
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output_coalesced));
    CHECK_CUDA(cudaFree(d_output_uncoalesced));
    delete[] h_input;
    delete[] h_expected_coalesced;
    delete[] h_expected_uncoalesced;
    delete[] h_output_coalesced;
    delete[] h_output_uncoalesced;

    return 0;
}
