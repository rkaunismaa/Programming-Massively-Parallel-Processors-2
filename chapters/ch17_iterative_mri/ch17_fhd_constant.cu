/*
 * ch17_fhd_constant.cu — Chapter 17: Iterative MRI Reconstruction
 * Fig 17.12 + Fig 17.13 + Fig 17.7: Constant memory for k-space data
 *
 * Adds constant memory for kx, ky, kz k-space coordinates. Since constant
 * memory is only 64KB, large datasets are chunked — the kernel is invoked
 * multiple times, each processing a chunk that fits in constant memory.
 * All threads in a warp access the same k-space element → ideal for
 * constant cache (96%+ cache hit rate, warp broadcast).
 *
 * Build: nvcc -std=c++17 -arch=sm_61 -O2 -o ch17_fhd_constant ch17_fhd_constant.cu
 * Run:   ./ch17_fhd_constant
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <string.h>

#include "../common/cuda_utils.cuh"

#define PI 3.14159265358979323846f
#define CHUNK_SIZE 4096  // Must fit 3 x CHUNK_SIZE x 4 bytes < 64KB

// ============================================================
// cmpMu kernel (Fig 17.7)
// ============================================================
#define MU_THREADS_PER_BLOCK 256

__global__ void cmpMu(const float* __restrict__ rPhi, const float* __restrict__ iPhi,
                      const float* __restrict__ rD,   const float* __restrict__ iD,
                      float* __restrict__ rMu, float* __restrict__ iMu, int M) {
    int m = blockIdx.x * MU_THREADS_PER_BLOCK + threadIdx.x;
    if (m >= M) return;
    rMu[m] = rPhi[m] * rD[m] + iPhi[m] * iD[m];
    iMu[m] = rPhi[m] * iD[m] - iPhi[m] * rD[m];
}

// ============================================================
// Constant memory for k-space chunks (64KB max)
// ============================================================
__constant__ float kx_c[CHUNK_SIZE];
__constant__ float ky_c[CHUNK_SIZE];
__constant__ float kz_c[CHUNK_SIZE];

// ============================================================
// cmpFhD constant kernel (Fig 17.13): Gather + registers + constant memory
//
// kx, ky, kz removed from parameter list — accessed from __constant__
// rMu, iMu still in global memory (2 accesses per iteration)
// Compute-to-memory ratio: ~1.63 OP/B
// ============================================================
#define FHD_THREADS_PER_BLOCK 256

__global__ void cmpFhD_constant(const float* __restrict__ x, const float* __restrict__ y,
                                const float* __restrict__ z,
                                const float* __restrict__ rMu, const float* __restrict__ iMu,
                                float* __restrict__ rFhD, float* __restrict__ iFhD,
                                int chunk_M, int N) {
    int n = blockIdx.x * FHD_THREADS_PER_BLOCK + threadIdx.x;
    if (n >= N) return;

    float xn_r = x[n];
    float yn_r = y[n];
    float zn_r = z[n];
    float rFhDn_r = rFhD[n];
    float iFhDn_r = iFhD[n];

    for (int m = 0; m < chunk_M; m++) {
        float expFhD = 2.0f * PI * (kx_c[m] * xn_r + ky_c[m] * yn_r + kz_c[m] * zn_r);
        float cArg = cosf(expFhD);
        float sArg = sinf(expFhD);
        rFhDn_r +=  rMu[m] * cArg - iMu[m] * sArg;
        iFhDn_r +=  iMu[m] * cArg + rMu[m] * sArg;
    }

    rFhD[n] = rFhDn_r;
    iFhD[n] = iFhDn_r;
}

// ============================================================
// CPU reference
// ============================================================
void cpu_fhd(const float* kx, const float* ky, const float* kz,
             const float* x, const float* y, const float* z,
             const float* rPhi, const float* iPhi,
             const float* rD, const float* iD,
             float* rFhD, float* iFhD, int M, int N) {
    for (int n = 0; n < N; n++) {
        rFhD[n] = 0.0f; iFhD[n] = 0.0f;
    }
    for (int m = 0; m < M; m++) {
        float rMu = rPhi[m] * rD[m] + iPhi[m] * iD[m];
        float iMu = rPhi[m] * iD[m] - iPhi[m] * rD[m];
        for (int n = 0; n < N; n++) {
            float expArg = 2.0f * PI * (kx[m] * x[n] + ky[m] * y[n] + kz[m] * z[n]);
            float cArg = cosf(expArg); float sArg = sinf(expArg);
            rFhD[n] += rMu * cArg - iMu * sArg;
            iFhD[n] += iMu * cArg + rMu * sArg;
        }
    }
}

int main() {
    CHECK_CUDA(cudaSetDevice(1));

    const int M = 2048;  // Large enough to exercise chunking
    const int N = 64;
    const size_t szM = M * sizeof(float);
    const size_t szN = N * sizeof(float);

    printf("=== Chapter 17: FHD Constant Memory Kernel (Fig 17.13 + 17.12 + 17.7) ===\n");
    printf("M (k-space samples) = %d, N (voxels) = %d, CHUNK_SIZE = %d\n", M, N, CHUNK_SIZE);

    float *kx_h = (float*)malloc(szM), *ky_h = (float*)malloc(szM), *kz_h = (float*)malloc(szM);
    float *x_h = (float*)malloc(szN), *y_h = (float*)malloc(szN), *z_h = (float*)malloc(szN);
    float *rPhi_h = (float*)malloc(szM), *iPhi_h = (float*)malloc(szM);
    float *rD_h = (float*)malloc(szM), *iD_h = (float*)malloc(szM);
    float *rFhD_h = (float*)malloc(szN), *iFhD_h = (float*)malloc(szN);
    float *rFhD_cpu = (float*)malloc(szN), *iFhD_cpu = (float*)malloc(szN);

    srand(42);
    for (int m = 0; m < M; m++) {
        kx_h[m] = (float)rand()/RAND_MAX - 0.5f;
        ky_h[m] = (float)rand()/RAND_MAX - 0.5f;
        kz_h[m] = (float)rand()/RAND_MAX - 0.5f;
        float phase = 2.0f * PI * (float)rand()/RAND_MAX;
        rPhi_h[m] = cosf(phase); iPhi_h[m] = sinf(phase);
        rD_h[m] = (float)rand()/RAND_MAX - 0.5f;
        iD_h[m] = (float)rand()/RAND_MAX - 0.5f;
    }
    for (int n = 0; n < N; n++) {
        x_h[n] = (float)(n % 4); y_h[n] = (float)((n / 4) % 4); z_h[n] = (float)(n / 16);
    }

    printf("\nComputing CPU reference...\n");
    gpu_timer cpu_timer;
    cpu_timer.start();
    cpu_fhd(kx_h, ky_h, kz_h, x_h, y_h, z_h, rPhi_h, iPhi_h, rD_h, iD_h,
            rFhD_cpu, iFhD_cpu, M, N);
    cpu_timer.stop();
    printf("  CPU time: %.3f ms\n", cpu_timer.elapsed_ms());

    // Device memory
    float *x_d, *y_d, *z_d;
    float *rPhi_d, *iPhi_d, *rD_d, *iD_d, *rMu_d, *iMu_d, *rFhD_d, *iFhD_d;
    CHECK_CUDA(cudaMalloc(&x_d, szN)); CHECK_CUDA(cudaMalloc(&y_d, szN));
    CHECK_CUDA(cudaMalloc(&z_d, szN));
    CHECK_CUDA(cudaMalloc(&rPhi_d, szM)); CHECK_CUDA(cudaMalloc(&iPhi_d, szM));
    CHECK_CUDA(cudaMalloc(&rD_d, szM)); CHECK_CUDA(cudaMalloc(&iD_d, szM));
    CHECK_CUDA(cudaMalloc(&rMu_d, szM)); CHECK_CUDA(cudaMalloc(&iMu_d, szM));
    CHECK_CUDA(cudaMalloc(&rFhD_d, szN)); CHECK_CUDA(cudaMalloc(&iFhD_d, szN));

    CHECK_CUDA(cudaMemcpy(x_d, x_h, szN, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(y_d, y_h, szN, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(z_d, z_h, szN, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(rPhi_d, rPhi_h, szM, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(iPhi_d, iPhi_h, szM, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(rD_d, rD_h, szM, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(iD_d, iD_h, szM, cudaMemcpyHostToDevice));

    int mu_blocks = (M + MU_THREADS_PER_BLOCK - 1) / MU_THREADS_PER_BLOCK;
    int fhd_blocks = (N + FHD_THREADS_PER_BLOCK - 1) / FHD_THREADS_PER_BLOCK;

    // Step 1: cmpMu
    cmpMu<<<mu_blocks, MU_THREADS_PER_BLOCK>>>(rPhi_d, iPhi_d, rD_d, iD_d, rMu_d, iMu_d, M);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // Step 2: cmpFhD with constant memory chunking
    printf("\nChunked constant-memory FHD kernel (%d chunks):\n", (M + CHUNK_SIZE - 1) / CHUNK_SIZE);

    // Initialize output
    CHECK_CUDA(cudaMemset(rFhD_d, 0, szN));
    CHECK_CUDA(cudaMemset(iFhD_d, 0, szN));

    gpu_timer gpu_timer;
    gpu_timer.start();

    for (int chunk_start = 0; chunk_start < M; chunk_start += CHUNK_SIZE) {
        int chunk_end = (chunk_start + CHUNK_SIZE < M) ? chunk_start + CHUNK_SIZE : M;
        int chunk_size = chunk_end - chunk_start;

        // Copy chunk of k-space data to constant memory
        CHECK_CUDA(cudaMemcpyToSymbol(kx_c, &kx_h[chunk_start], chunk_size * sizeof(float), 0,
                                      cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpyToSymbol(ky_c, &ky_h[chunk_start], chunk_size * sizeof(float), 0,
                                      cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpyToSymbol(kz_c, &kz_h[chunk_start], chunk_size * sizeof(float), 0,
                                      cudaMemcpyHostToDevice));

        // Launch kernel for this chunk (rMu/iMu offset by chunk_start)
        cmpFhD_constant<<<fhd_blocks, FHD_THREADS_PER_BLOCK>>>(
            x_d, y_d, z_d, &rMu_d[chunk_start], &iMu_d[chunk_start],
            rFhD_d, iFhD_d, chunk_size, N);
        CHECK_CUDA(cudaGetLastError());
    }

    CHECK_CUDA(cudaDeviceSynchronize());
    gpu_timer.stop();
    float gpu_ms = gpu_timer.elapsed_ms();

    CHECK_CUDA(cudaMemcpy(rFhD_h, rFhD_d, szN, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(iFhD_h, iFhD_d, szN, cudaMemcpyDeviceToHost));

    printf("  GPU cmpFhD time (all chunks): %.3f ms\n", gpu_ms);

    printf("\nValidating against CPU reference...\n");
    float tol = 1e-4f;
    bool pass_r = cpu_allclose(rFhD_h, rFhD_cpu, N, tol);
    bool pass_i = cpu_allclose(iFhD_h, iFhD_cpu, N, tol);

    if (pass_r && pass_i) {
        printf("  Validation: PASS (both real and imaginary components)\n");
    } else {
        if (!pass_r) printf("  FAIL: Real component mismatch\n");
        if (!pass_i) printf("  FAIL: Imaginary component mismatch\n");
        for (int i = 0; i < 5 && i < N; i++)
            printf("  [%d] real: gpu=%.6f cpu=%.6f  imag: gpu=%.6f cpu=%.6f\n",
                   i, rFhD_h[i], rFhD_cpu[i], iFhD_h[i], iFhD_cpu[i]);
    }

    cudaFree(x_d); cudaFree(y_d); cudaFree(z_d);
    cudaFree(rPhi_d); cudaFree(iPhi_d); cudaFree(rD_d); cudaFree(iD_d);
    cudaFree(rMu_d); cudaFree(iMu_d); cudaFree(rFhD_d); cudaFree(iFhD_d);
    free(kx_h); free(ky_h); free(kz_h); free(x_h); free(y_h); free(z_h);
    free(rPhi_h); free(iPhi_h); free(rD_h); free(iD_h);
    free(rFhD_h); free(iFhD_h); free(rFhD_cpu); free(iFhD_cpu);

    printf("\nDone.\n");
    return (pass_r && pass_i) ? 0 : 1;
}
