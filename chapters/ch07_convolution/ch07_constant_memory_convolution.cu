/*
 * =============================================================================
 *  Chapter 7: Convolution — Constant Memory for Filter (Fig 7.9)
 * =============================================================================
 *  Book:      Programming Massively Parallel Processors (Kirk, Hwu & El Hajj,
 *             4th Edition, Morgan Kaufmann 2023)
 *  Figure:    7.9 — Convolution kernel using __constant__ memory for F
 *  Purpose:   Demonstrate constant memory caching benefit. When all threads
 *             access the same filter element simultaneously (broadcast),
 *             the constant cache serves it in one transaction, eliminating
 *             DRAM bandwidth usage for the filter.
 *  Hardware:  GTX 1050, sm_61 (Pascal)
 * =============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../common/cuda_utils.cuh"

#define FILTER_RADIUS 2

// Constant memory declaration for the filter (Fig 7.9)
__constant__ float F_c[2*FILTER_RADIUS+1][2*FILTER_RADIUS+1];

// Kernel identical to Fig 7.7 but accesses F via constant memory symbol
__global__ void convolution_2D_basic_constant_kernel(
    float *N, float *P, int r, int width, int height) {
    int outCol = blockIdx.x * blockDim.x + threadIdx.x;
    int outRow = blockIdx.y * blockDim.y + threadIdx.y;
    float Pvalue = 0.0f;

    for (int fRow = 0; fRow < 2 * r + 1; fRow++) {
        for (int fCol = 0; fCol < 2 * r + 1; fCol++) {
            int inRow = outRow - r + fRow;
            int inCol = outCol - r + fCol;
            if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width) {
                Pvalue += F_c[fRow][fCol] * N[inRow * width + inCol];
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

    printf("=========================================================\n");
    printf("Chapter 7: Constant Memory Convolution (Fig 7.9)\n");
    printf("Image size: %d x %d, Filter radius: %d (filter %dx%d)\n",
           width, height, r, fsz, fsz);

    // Allocate host memory
    float *N_h = (float*)malloc(n * sizeof(float));
    float *F_h = (float*)malloc(fsz * fsz * sizeof(float));
    float *P_h = (float*)malloc(n * sizeof(float));
    float *P_cpu = (float*)malloc(n * sizeof(float));

    // Initialize
    for (int i = 0; i < n; i++) {
        N_h[i] = fmodf(i * 0.01f, 10.0f);
    }
    for (int fy = 0; fy < fsz; fy++) {
        for (int fx = 0; fx < fsz; fx++) {
            float dx = fx - r, dy = fy - r;
            F_h[fy * fsz + fx] = expf(-(dx*dx+dy*dy) / 2.0f);
        }
    }

    // Allocate device image memory (F goes to constant memory)
    float *d_N, *d_P;
    CHECK_CUDA(cudaMalloc(&d_N, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_P, n * sizeof(float)));

    // Copy image to device
    CHECK_CUDA(cudaMemcpy(d_N, N_h, n * sizeof(float), cudaMemcpyHostToDevice));

    // Copy filter to constant memory using cudaMemcpyToSymbol (Fig 7.9)
    size_t filter_bytes = fsz * fsz * sizeof(float);
    CHECK_CUDA(cudaMemcpyToSymbol(F_c, F_h, filter_bytes));

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);

    printf("Grid: %d x %d, Block: %d x %d\n",
           grid.x, grid.y, block.x, block.y);

    // Warmup
    convolution_2D_basic_constant_kernel<<<grid, block>>>(d_N, d_P, r, width, height);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Timed run
    gpu_timer timer;
    timer.start();
    convolution_2D_basic_constant_kernel<<<grid, block>>>(d_N, d_P, r, width, height);
    timer.stop();
    float elapsed = timer.elapsed_ms();

    CHECK_CUDA(cudaMemcpy(P_h, d_P, n * sizeof(float), cudaMemcpyDeviceToHost));

    // CPU reference (flatten filter to 1D for comparison)
    cpu_convolve(N_h, F_h, P_cpu, r, width, height);

    bool passed = cpu_allclose(P_h, P_cpu, n, 1e-2f);
    printf("\n---------------------------------------------------------\n");
    printf("Kernel time: %.2f ms\n", elapsed);
    // With constant memory, filter bandwidth is essentially eliminated
    float total_bytes_loaded = (float)n * fsz * fsz * sizeof(float);
    printf("Effective BW (image only): %.2f GB/s\n",
           (float)n * fsz * fsz * sizeof(float) / (elapsed * 1e6));
    printf("%s\n", passed ? "Validation: PASSED" : "Validation: FAILED");

    CHECK_CUDA(cudaFree(d_N));
    CHECK_CUDA(cudaFree(d_P));
    free(N_h); free(F_h); free(P_h); free(P_cpu);
    return 0;
}
