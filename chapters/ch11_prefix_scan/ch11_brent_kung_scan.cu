/*
 * Fig 11.7 — Brent-Kung Parallel Inclusive Scan (Segmented)
 *
 * Uses a reduction tree phase (N-1 ops) + reverse tree phase (N-1-log2(N) ops).
 * Total work: 2N - 2 - log2(N) ops = O(N) — work-efficient!
 *
 * Each block has SECTION_SIZE/2 threads and processes SECTION_SIZE elements
 * (each thread loads/outputs 2 elements). SECTION_SIZE=2048 max on GTX 1050.
 *
 * The reduction tree is careful to maintain the right intermediate values
 * for the reverse tree phase. Uses a convergent thread-to-data mapping
 * via: index = (threadIdx.x+1)*2*stride - 1.
 *
 * Hardware: GTX 1050 (sm_61 Pascal, device 1)
 * Compile:  nvcc -std=c++17 -arch=sm_61 -O2 -o ch11_brent_kung_scan ch11_brent_kung_scan.cu
 */

#include "../common/cuda_utils.cuh"
#include <cstdlib>
#include <ctime>

// SECTION_SIZE = 2 * blockDim.x (each thread handles 2 elements)
// Max threads/block = 1024 → max SECTION_SIZE = 2048
#define SECTION_SIZE 2048

// ---------------------------------------------------------------------------
// Fig 11.7 — Brent-Kung Inclusive Scan Kernel
//
// Reduction tree phase: builds partial sums at positions k*2^n-1
// Reverse tree phase:  distributes accumulated sums to remaining positions
// ---------------------------------------------------------------------------
__global__ void brent_kung_scan_kernel(const float *X, float *Y, unsigned int N) {
    __shared__ float XY[SECTION_SIZE];

    unsigned int base = 2 * blockIdx.x * blockDim.x;
    unsigned int i = base + threadIdx.x;

    // Each thread loads two elements (coalesced: adjacent threads load adjacent elements)
    if (i < N) {
        XY[threadIdx.x] = X[i];
    } else {
        XY[threadIdx.x] = 0.0f;
    }

    if (i + blockDim.x < N) {
        XY[threadIdx.x + blockDim.x] = X[i + blockDim.x];
    } else if (i < N || (base + threadIdx.x + blockDim.x) < N) {
        // Only zero if this actually corresponds to a real element
        XY[threadIdx.x + blockDim.x] = 0.0f;
    }

    // ---------------------------------------------------------------
    // Phase 1: Reduction tree (N-1 operations)
    //   stride: 1, 2, 4, ..., blockDim.x (S=2^n)
    //   index = (threadIdx.x+1)*2*stride - 1 → positions k*2*stride-1
    // ---------------------------------------------------------------
    for (unsigned int stride = 1; stride <= blockDim.x; stride *= 2) {
        __syncthreads();

        unsigned int index = (threadIdx.x + 1) * 2 * stride - 1;
        if (index < SECTION_SIZE) {
            XY[index] += XY[index - stride];
        }
    }

    // ---------------------------------------------------------------
    // Phase 2: Reverse (distribution) tree (N-1-log2(N) operations)
    //   stride: SECTION_SIZE/4, SECTION_SIZE/8, ..., 1
    //   index = (threadIdx.x+1)*stride*2 - 1 (same formula)
    //   Pushes values from source positions stride to the right
    // ---------------------------------------------------------------
    for (int stride = SECTION_SIZE / 4; stride > 0; stride /= 2) {
        __syncthreads();

        unsigned int index = (threadIdx.x + 1) * stride * 2 - 1;
        if (index + stride < SECTION_SIZE) {
            XY[index + stride] += XY[index];
        }
    }

    __syncthreads();

    // Write results
    if (i < N) {
        Y[i] = XY[threadIdx.x];
    }
    if (i + blockDim.x < N) {
        Y[i + blockDim.x] = XY[threadIdx.x + blockDim.x];
    }
}

// ---------------------------------------------------------------------------
// Host-side sequential inclusive scan (segment-local)
// Each SECTION_SIZE segment is scanned independently (segmented scan).
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

    // Problem size: use a multiple of SECTION_SIZE to exercise multiple blocks
    const unsigned int N = 4 * SECTION_SIZE;  // 8192 elements, 4 blocks

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

    // Section_size/2 threads per block (each thread processes 2 elements)
    unsigned int threads_per_block = SECTION_SIZE / 2;  // 1024
    unsigned int blocks = (N + SECTION_SIZE - 1) / SECTION_SIZE;

    dim3 grid(blocks, 1, 1);
    dim3 block(threads_per_block, 1, 1);

    // Warm-up
    brent_kung_scan_kernel<<<grid, block>>>(d_X, d_Y, N);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Timed run
    gpu_timer timer;
    timer.start();
    brent_kung_scan_kernel<<<grid, block>>>(d_X, d_Y, N);
    timer.stop();
    float ms = timer.elapsed_ms();

    CHECK_CUDA(cudaMemcpy(h_Y_gpu, d_Y, N * sizeof(float), cudaMemcpyDeviceToHost));

    bool passed = cpu_allclose(h_Y_ref, h_Y_gpu, N, 1e-4f);

    float bytes = (float)N * sizeof(float) * 2.0f;
    float bw = bytes / (ms * 1e-3f) / 1.0e9f;

    printf("\n");
    printf("---------------------------------------------------------\n");
    printf("Brent-Kung Inclusive Scan (segmented, SECTION_SIZE=%d)\n", SECTION_SIZE);
    printf("---------------------------------------------------------\n");
    printf("  Input size    : %u elements\n", N);
    printf("  Blocks        : %u\n", blocks);
    printf("  Threads/block : %d\n", threads_per_block);
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
