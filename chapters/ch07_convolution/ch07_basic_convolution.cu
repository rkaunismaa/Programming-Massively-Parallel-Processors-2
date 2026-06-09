/*
 * =============================================================================
 *  Chapter 7: Convolution — Basic Parallel 2D Convolution
 * =============================================================================
 *  Book:      Programming Massively Parallel Processors (Kirk, Hwu & El Hajj,
 *             4th Edition, Morgan Kaufmann 2023)
 *  Figure:    7.7 — Basic parallel convolution kernel
 *  Purpose:   Naive parallel convolution where each thread computes one output
 *             pixel by iterating over the filter/window neighborhood.
 *  Hardware:  GTX 1050, sm_61 (Pascal) — all code targets this GPU
 * =============================================================================
 *
 *  BASIC CONVOLUTION CONCEPT
 *  ---------------------------------------------------------------------------
 *  Each thread is mapped to one output element P[row][col]. The thread
 *  iterates over the filter radius r in both dimensions, accumulates
 *  weighted sums from input array N, and handles boundary conditions
 *  (ghost cells) by treating out-of-bounds accesses as zero.
 *
 *  Memory access pattern: for each output element, the kernel reads
 *  (2r+1)^2 elements from global memory N plus (2r+1) filter coefficients.
 *  This is extremely memory-bandwidth intensive — arithmetic intensity
 *  is only ~0.25 OP/B (2 operations per 8 bytes loaded).
 * =============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include "../common/cuda_utils.cuh"

int FILTER_RADIUS = 2;

// Figure 7.7 — Basic parallel convolution kernel
__global__ void convolution_2D_basic_kernel(
    float *N, float *F, float *P, int r, int width, int height) {
    int outCol = blockIdx.x * blockDim.x + threadIdx.x;
    int outRow = blockIdx.y * blockDim.y + threadIdx.y;
    float Pvalue = 0.0f;

    for (int fRow = 0; fRow < 2 * r + 1; fRow++) {
        for (int fCol = 0; fCol < 2 * r + 1; fCol++) {
            int inRow = outRow - r + fRow;
            int inCol = outCol - r + fCol;
            if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width) {
                Pvalue += F[fRow * (2*r+1) + fCol] * N[inRow * width + inCol];
            }
        }
    }
    P[outRow * width + outCol] = Pvalue;
}

// CPU reference for validation
void cpu_convolve(const float* N, const float* F, float* P,
                  int r, int width, int height) {
    int fsz = 2*r+1;
    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            float Pvalue = 0.0f;
            for (int fRow = 0; fRow < fsz; fRow++) {
                for (int fCol = 0; fCol < fsz; fCol++) {
                    int inRow = row - r + fRow;
                    int inCol = col - r + fCol;
                    if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width) {
                        Pvalue += F[fRow * fsz + fCol] * N[inRow * width + inCol];
                    }
                }
            }
            P[row * width + col] = Pvalue;
        }
    }
}

int main() {
    cudaSetDevice(1);
    print_device_info(1);

    int r = FILTER_RADIUS;
    int fsz = 2*r+1;
    int width = 1024;
    int height = 1024;
    int n = width * height;
    float filter_size_bytes = fsz * fsz * sizeof(float);

    printf("=========================================================\n");
    printf("Chapter 7: Basic Parallel 2D Convolution (Fig 7.7)\n");
    printf("Image size: %d x %d, Filter radius: %d (filter %dx%d)\n",
           width, height, r, fsz, fsz);

    // Allocate host memory
    float *N_h = (float*)malloc(n * sizeof(float));
    float *F_h = (float*)malloc(fsz * fsz * sizeof(float));
    float *P_h = (float*)malloc(n * sizeof(float));
    float *P_cpu = (float*)malloc(n * sizeof(float));

    // Initialize: image with sequential values, filter as Gaussian-like
    for (int i = 0; i < n; i++) {
        N_h[i] = fmodf(i * 0.01f, 10.0f);
    }
    // Simple averaging/smoothing-ish filter
    float weights[] = {1, 2, 1, 2, 4, 2, 1, 2, 1}; // 3x3 sum=16 (not used for r=2)
    for (int i = 0; i < fsz*fsz; i++) {
        int fx = fmod(i, fsz);
        int fy = i / fsz;
        // Gaussian-like weights
        float dx = fx - r, dy = fy - r;
        F_h[i] = expf(-(dx*dx+dy*dy) / 2.0f);
    }

    // Allocate device memory
    float *d_N, *d_F, *d_P;
    CHECK_CUDA(cudaMalloc(&d_N, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_F, fsz * fsz * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_P, n * sizeof(float)));

    // Copy to device
    CHECK_CUDA(cudaMemcpy(d_N, N_h, n * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_F, F_h, fsz * fsz * sizeof(float), cudaMemcpyHostToDevice));

    // Launch configuration (4x4 thread blocks for simplicity)
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);

    printf("Grid: %d x %d, Block: %d x %d\n",
           grid.x, grid.y, block.x, block.y);

    // Warmup
    convolution_2D_basic_kernel<<<grid, block>>>(d_N, d_F, d_P, r, width, height);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Timed run
    gpu_timer timer;
    timer.start();
    convolution_2D_basic_kernel<<<grid, block>>>(d_N, d_F, d_P, r, width, height);
    timer.stop();
    float elapsed = timer.elapsed_ms();

    // Copy result back
    CHECK_CUDA(cudaMemcpy(P_h, d_P, n * sizeof(float), cudaMemcpyDeviceToHost));

    // CPU reference
    cpu_convolve(N_h, F_h, P_cpu, r, width, height);

    // Validation
    bool passed = cpu_allclose(P_h, P_cpu, n, 1e-2f);
    printf("\n---------------------------------------------------------\n");
    printf("Kernel time: %.2f ms\n", elapsed);
    float total_ops = 2.0f * fsz * fsz * n; // multiply + add per filter element per output
    float total_bytes_loaded = (float)n * fsz * fsz * sizeof(float);
    printf("Effective bandwidth: %.2f GB/s\n", total_bytes_loaded / (elapsed * 1e6));
    printf("%s\n", passed ? "Validation: PASSED" : "Validation: FAILED");

    // Cleanup
    CHECK_CUDA(cudaFree(d_N));
    CHECK_CUDA(cudaFree(d_F));
    CHECK_CUDA(cudaFree(d_P));
    free(N_h); free(F_h); free(P_h); free(P_cpu);

    return 0;
}
