/*
 * Section 11.5 / Fig 11.8 — Three-Phase Coarsened Parallel Scan
 *
 * Improves work efficiency by having each thread perform a sequential scan
 * on its own contiguous subsection before the block-wide parallel scan.
 *
 * Three phases (per Fig 11.8):
 *   1. Coalesced load into shared mem, then each thread sequentially scans
 *      its contiguous CFACTOR-element subsection
 *   2. Block-wide Kogge-Stone scan on the last element of each subsection
 *   3. Each thread adds predecessor's last-element sum to its own elements
 *
 * Data layout in shared memory:
 *   XY[tid * CFACTOR + c] = element c of thread tid's contiguous subsection
 *   This gives each thread CFACTOR CONTIGUOUS elements for sequential scan.
 *
 * Hardware: GTX 1050 (sm_61 Pascal, device 1)
 * Compile:  nvcc -std=c++17 -arch=sm_61 -O2 -o ch11_coarsened_scan ch11_coarsened_scan.cu
 */

#include "../common/cuda_utils.cuh"
#include <cstdlib>
#include <ctime>

#define T 1024       // threads per block
#define CFACTOR 4    // coarsening factor (elements per thread)
#define SECTION_SIZE (T * CFACTOR)  // 4096 elements per block

// ---------------------------------------------------------------------------
// Three-phase coarsened scan kernel
// ---------------------------------------------------------------------------
__global__ void coarsened_scan_kernel(const float *X, float *Y, unsigned int N) {
    __shared__ float XY[SECTION_SIZE];  // shared memory for the entire section

    unsigned int block_start = blockIdx.x * SECTION_SIZE;
    unsigned int tid = threadIdx.x;

    // ---------------------------------------------------------------
    // PHASE 0: Cooperative coalesced load into shared memory
    //   Adjacent threads load adjacent elements from global memory.
    //   XY is laid out in DATA ORDER: XY[offset] = X[block_start + offset]
    // ---------------------------------------------------------------
    for (unsigned int offset = tid; offset < SECTION_SIZE; offset += T) {
        unsigned int idx = block_start + offset;
        if (idx < N) {
            XY[offset] = X[idx];
        } else {
            XY[offset] = 0.0f;
        }
    }
    __syncthreads();

    // ---------------------------------------------------------------
    // PHASE 1: Each thread performs a sequential scan on its own
    //   CONTIGUOUS subsection of CFACTOR elements.
    //   Thread tid's subsection: XY[tid*CFACTOR .. tid*CFACTOR+CFACTOR-1]
    // ---------------------------------------------------------------
    {
        unsigned int start = tid * CFACTOR;
        float acc = 0.0f;
        for (unsigned int c = 0; c < CFACTOR; c++) {
            acc += XY[start + c];
            XY[start + c] = acc;
        }
    }
    __syncthreads();

    // ---------------------------------------------------------------
    // PHASE 2: Block-wide Kogge-Stone scan on last elements
    //   last_s[tid] = XY[tid*CFACTOR + CFACTOR-1] (last element of each
    //   thread's subsection — the sum of that thread's CFACTOR elements)
    // ---------------------------------------------------------------
    __shared__ float last_s[T];
    last_s[tid] = XY[tid * CFACTOR + CFACTOR - 1];
    __syncthreads();

    for (unsigned int stride = 1; stride < T; stride *= 2) {
        __syncthreads();
        float temp;
        if (tid >= stride) {
            temp = last_s[tid] + last_s[tid - stride];
        }
        __syncthreads();
        if (tid >= stride) {
            last_s[tid] = temp;
        }
    }
    __syncthreads();

    // ---------------------------------------------------------------
    // PHASE 3: Add predecessor's accumulated sum
    //   Thread t adds last_s[t-1] (sum of all elements before this
    //   thread's subsection) to ALL its subsection elements.
    //   The last element gets the correct cumulative value because
    //   last_s[tid] already has the cumulative total from Phase 2.
    // ---------------------------------------------------------------
    float add_val = (tid > 0) ? last_s[tid - 1] : 0.0f;
    {
        unsigned int start = tid * CFACTOR;
        for (unsigned int c = 0; c < CFACTOR; c++) {
            XY[start + c] += add_val;
        }
    }
    __syncthreads();

    // ---------------------------------------------------------------
    // Write results back to global memory (coalesced: contiguous write)
    // ---------------------------------------------------------------
    for (unsigned int offset = tid; offset < SECTION_SIZE; offset += T) {
        unsigned int idx = block_start + offset;
        if (idx < N) {
            Y[idx] = XY[offset];
        }
    }
}

// ---------------------------------------------------------------------------
// Host-side segment-local reference
// ---------------------------------------------------------------------------
void host_segmented_scan(const float *X, float *Y, unsigned int N, unsigned int seg_size) {
    for (unsigned int s = 0; s < N; s += seg_size) {
        unsigned int end = (s + seg_size < N) ? s + seg_size : N;
        Y[s] = X[s];
        for (unsigned int i = s + 1; i < end; i++) {
            Y[i] = Y[i - 1] + X[i];
        }
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main() {
    int dev = 1;
    cudaSetDevice(dev);
    print_device_info(dev);

    const unsigned int N = 4 * SECTION_SIZE;  // 16384 elements, 4 blocks

    float *h_X = new float[N];
    float *h_Y_gpu = new float[N];
    float *h_Y_ref = new float[N];

    srand(time(nullptr));
    for (unsigned int i = 0; i < N; i++) {
        h_X[i] = (float)(rand() % 100) / 10.0f;
    }

    host_segmented_scan(h_X, h_Y_ref, N, SECTION_SIZE);

    float *d_X, *d_Y;
    CHECK_CUDA(cudaMalloc(&d_X, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_Y, N * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_X, h_X, N * sizeof(float), cudaMemcpyHostToDevice));

    unsigned int blocks = (N + SECTION_SIZE - 1) / SECTION_SIZE;

    // Warm-up
    coarsened_scan_kernel<<<blocks, T>>>(d_X, d_Y, N);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Restore input (coarsened scan does not modify d_X)
    CHECK_CUDA(cudaMemcpy(d_X, h_X, N * sizeof(float), cudaMemcpyHostToDevice));

    // Timed run
    gpu_timer timer;
    timer.start();
    coarsened_scan_kernel<<<blocks, T>>>(d_X, d_Y, N);
    timer.stop();
    float ms = timer.elapsed_ms();

    CHECK_CUDA(cudaMemcpy(h_Y_gpu, d_Y, N * sizeof(float), cudaMemcpyDeviceToHost));

    bool passed = cpu_allclose(h_Y_ref, h_Y_gpu, N, 1e-4f);

    float bytes = (float)N * sizeof(float) * 2.0f;
    float bw = bytes / (ms * 1e-3f) / 1.0e9f;

    printf("\n");
    printf("---------------------------------------------------------\n");
    printf("Coarsened Scan (CFACTOR=%d, T=%d, SECTION_SIZE=%d)\n", CFACTOR, T, SECTION_SIZE);
    printf("---------------------------------------------------------\n");
    printf("  Input size    : %u elements\n", N);
    printf("  Blocks        : %u\n", blocks);
    printf("  Threads/block : %d\n", T);
    printf("  Elements/block: %d\n", SECTION_SIZE);
    printf("  Kernel time   : %.3f ms\n", ms);
    printf("  Bandwidth     : %.2f GB/s\n", bw);
    printf("  Validation    : %s\n", passed ? "PASSED" : "FAILED");
    printf("---------------------------------------------------------\n");

    printf("\nFirst 12 results:\n");
    for (unsigned int i = 0; i < 12 && i < N; i++) {
        printf("  Y[%2u] = %.2f (expected %.2f)%s\n",
               i, h_Y_gpu[i], h_Y_ref[i],
               std::abs(h_Y_gpu[i] - h_Y_ref[i]) > 1e-4f ? " MISMATCH!" : "");
    }

    delete[] h_X;
    delete[] h_Y_gpu;
    delete[] h_Y_ref;
    CHECK_CUDA(cudaFree(d_X));
    CHECK_CUDA(cudaFree(d_Y));

    return passed ? 0 : 1;
}
