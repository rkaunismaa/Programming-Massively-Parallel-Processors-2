/*
 * Fig 11.3 — Kogge-Stone Parallel Inclusive Scan (Segmented)
 *
 * Each thread block independently scans a SECTION_SIZE segment of the input.
 * Uses the Kogge-Stone adder design: stride-doubling pattern with a temp
 * variable and extra __syncthreads() to avoid write-after-read race conditions.
 *
 * Key differences from reduction (Ch10):
 *   - Each thread accumulates partial sums for its OWN position, not a tree root
 *   - Temp variable + double barrier needed because updated values would be read
 *     by other threads in the same iteration (write-after-read hazard)
 *   - Work complexity: O(N log N) vs sequential O(N) — not work-efficient
 *
 * Hardware: GTX 1050 (sm_61 Pascal, device 1)
 * Compile:  nvcc -std=c++17 -arch=sm_61 -O2 -o ch11_kogge_stone_scan ch11_kogge_stone_scan.cu
 */

#include "../common/cuda_utils.cuh"
#include <cstdlib>
#include <ctime>

#define SECTION_SIZE 1024

// ---------------------------------------------------------------------------
// Fig 11.3 — Kogge-Stone Inclusive Scan Kernel
// ---------------------------------------------------------------------------
// Each thread loads one element into shared memory, then iterates with
// stride-doubling. At each step, threadIdx.x >= stride reads XY[threadIdx.x-stride]
// and adds it to its own value. Temp variable + extra barrier avoids WAW hazard.
// ---------------------------------------------------------------------------
__global__ void kogge_stone_scan_kernel(const float *X, float *Y, unsigned int N) {
    __shared__ float XY[SECTION_SIZE];

    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load input into shared memory (zero-pad beyond N)
    if (i < N) {
        XY[threadIdx.x] = X[i];
    } else {
        XY[threadIdx.x] = 0.0f;
    }

    // Kogge-Stone scan loop (stride doubling)
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

    // Write result
    if (i < N) {
        Y[i] = XY[threadIdx.x];
    }
}

// ---------------------------------------------------------------------------
// Host-side sequential inclusive scan (segment-local, per book Fig 11.3)
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
    // Select GTX 1050 (device 1)
    int dev = 1;
    cudaSetDevice(dev);
    print_device_info(dev);

    // Problem size: use a multiple of SECTION_SIZE that exercises multiple blocks
    const unsigned int N = 4 * SECTION_SIZE;  // 4096 elements, 4 blocks

    // Allocate host memory
    float *h_X = new float[N];
    float *h_Y_gpu = new float[N];
    float *h_Y_ref = new float[N];

    // Initialize input with random values
    srand(time(nullptr));
    for (unsigned int i = 0; i < N; i++) {
        h_X[i] = (float)(rand() % 100) / 10.0f;  // 0.0 to 9.9
    }

    // Compute reference on CPU
    host_segmented_scan(h_X, h_Y_ref, N, SECTION_SIZE);

    // Allocate device memory
    float *d_X, *d_Y;
    CHECK_CUDA(cudaMalloc(&d_X, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_Y, N * sizeof(float)));

    // Copy input to device
    CHECK_CUDA(cudaMemcpy(d_X, h_X, N * sizeof(float), cudaMemcpyHostToDevice));

    // Launch Kogge-Stone scan kernel (one block per SECTION_SIZE segment)
    unsigned int blocks = (N + SECTION_SIZE - 1) / SECTION_SIZE;
    dim3 grid(blocks, 1, 1);
    dim3 block(SECTION_SIZE, 1, 1);

    // Warm-up
    kogge_stone_scan_kernel<<<grid, block>>>(d_X, d_Y, N);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Timed run
    gpu_timer timer;
    timer.start();
    kogge_stone_scan_kernel<<<grid, block>>>(d_X, d_Y, N);
    timer.stop();
    float ms = timer.elapsed_ms();

    // Copy result back
    CHECK_CUDA(cudaMemcpy(h_Y_gpu, d_Y, N * sizeof(float), cudaMemcpyDeviceToHost));

    // Validate
    bool passed = cpu_allclose(h_Y_ref, h_Y_gpu, N, 1e-4f);

    // Calculate bandwidth (read N floats, write N floats)
    float bytes = (float)N * sizeof(float) * 2.0f;
    float bw = bytes / (ms * 1e-3f) / 1.0e9f;  // GB/s

    // Print results
    printf("\n");
    printf("---------------------------------------------------------\n");
    printf("Kogge-Stone Inclusive Scan (segmented, SECTION_SIZE=%d)\n", SECTION_SIZE);
    printf("---------------------------------------------------------\n");
    printf("  Input size  : %u elements\n", N);
    printf("  Blocks      : %u\n", blocks);
    printf("  Threads/block: %d\n", SECTION_SIZE);
    printf("  Kernel time : %.3f ms\n", ms);
    printf("  Bandwidth   : %.2f GB/s\n", bw);
    printf("  Validation  : %s\n", passed ? "PASSED" : "FAILED");
    printf("---------------------------------------------------------\n");

    // Show first 16 scan results as a sanity check
    printf("\nFirst 12 results:\n");
    for (unsigned int i = 0; i < 12 && i < N; i++) {
        printf("  Y[%2u] = %.2f (expected %.2f)%s\n",
               i, h_Y_gpu[i], h_Y_ref[i],
               std::abs(h_Y_gpu[i] - h_Y_ref[i]) > 1e-4f ? " MISMATCH!" : "");
    }

    // Clean up
    delete[] h_X;
    delete[] h_Y_gpu;
    delete[] h_Y_ref;
    CHECK_CUDA(cudaFree(d_X));
    CHECK_CUDA(cudaFree(d_Y));

    return passed ? 0 : 1;
}
