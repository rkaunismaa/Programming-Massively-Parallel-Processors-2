/*
 * ch17_fhd_scatter.cu — Chapter 17: Iterative MRI Reconstruction
 * Fig 17.5 + Fig 17.7: Scatter approach to FHD computation
 *
 * Each thread processes one k-space sample and scatters its contribution
 * to ALL voxels using atomicAdd. Demonstrates why scatter is suboptimal.
 *
 * Build: nvcc -std=c++17 -arch=sm_61 -O2 -o ch17_fhd_scatter ch17_fhd_scatter.cu
 * Run:   ./ch17_fhd_scatter
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>

#include "../common/cuda_utils.cuh"

#define PI 3.14159265358979323846f

// ============================================================
// cmpMu kernel (Fig 17.7): Compute rMu and iMu
// One thread per k-space sample
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
// cmpFhD scatter kernel (Fig 17.5): One thread per k-space sample
// Each thread scatters to ALL voxels — needs atomicAdd
// ============================================================
#define FHD_THREADS_PER_BLOCK 256

__global__ void cmpFhD_scatter(const float* __restrict__ kx, const float* __restrict__ ky,
                               const float* __restrict__ kz,
                               const float* __restrict__ x, const float* __restrict__ y,
                               const float* __restrict__ z,
                               const float* __restrict__ rMu, const float* __restrict__ iMu,
                               float* __restrict__ rFhD, float* __restrict__ iFhD,
                               int M, int N) {
    int m = blockIdx.x * FHD_THREADS_PER_BLOCK + threadIdx.x;
    if (m >= M) return;

    float kxm = kx[m], kym = ky[m], kzm = kz[m];
    float rMum = rMu[m], iMum = iMu[m];

    for (int n = 0; n < N; n++) {
        float expFhD = 2.0f * PI * (kxm * x[n] + kym * y[n] + kzm * z[n]);
        float cArg = cosf(expFhD);
        float sArg = sinf(expFhD);
        atomicAdd(&rFhD[n],  rMum * cArg - iMum * sArg);
        atomicAdd(&iFhD[n],  iMum * cArg + rMum * sArg);
    }
}

// ============================================================
// CPU reference: computes FHD sequentially
// ============================================================
void cpu_fhd(const float* kx, const float* ky, const float* kz,
             const float* x, const float* y, const float* z,
             const float* rPhi, const float* iPhi,
             const float* rD, const float* iD,
             float* rFhD, float* iFhD, int M, int N) {
    for (int n = 0; n < N; n++) {
        rFhD[n] = 0.0f;
        iFhD[n] = 0.0f;
    }
    for (int m = 0; m < M; m++) {
        float rMu = rPhi[m] * rD[m] + iPhi[m] * iD[m];
        float iMu = rPhi[m] * iD[m] - iPhi[m] * rD[m];
        for (int n = 0; n < N; n++) {
            float expArg = 2.0f * PI * (kx[m] * x[n] + ky[m] * y[n] + kz[m] * z[n]);
            float cArg = cosf(expArg);
            float sArg = sinf(expArg);
            rFhD[n] += rMu * cArg - iMu * sArg;
            iFhD[n] += iMu * cArg + rMu * sArg;
        }
    }
}

int main() {
    CHECK_CUDA(cudaSetDevice(1));

    // Small test: M k-space samples, N voxels (e.g., 4x4x4 = 64)
    const int M = 512;
    const int N = 64;  // 4x4x4 voxels

    const size_t szM = M * sizeof(float);
    const size_t szN = N * sizeof(float);

    printf("=== Chapter 17: FHD Scatter Kernel (Fig 17.5 + 17.7) ===\n");
    printf("M (k-space samples) = %d, N (voxels) = %d\n", M, N);

    // Allocate host memory
    float *kx_h = (float*)malloc(szM), *ky_h = (float*)malloc(szM), *kz_h = (float*)malloc(szM);
    float *x_h = (float*)malloc(szN), *y_h = (float*)malloc(szN), *z_h = (float*)malloc(szN);
    float *rPhi_h = (float*)malloc(szM), *iPhi_h = (float*)malloc(szM);
    float *rD_h = (float*)malloc(szM), *iD_h = (float*)malloc(szM);
    float *rFhD_h = (float*)malloc(szN), *iFhD_h = (float*)malloc(szN);
    float *rFhD_cpu = (float*)malloc(szN), *iFhD_cpu = (float*)malloc(szN);

    // Initialize with synthetic data
    srand(42);
    for (int m = 0; m < M; m++) {
        kx_h[m] = (float)rand()/RAND_MAX - 0.5f;
        ky_h[m] = (float)rand()/RAND_MAX - 0.5f;
        kz_h[m] = (float)rand()/RAND_MAX - 0.5f;
        float phase = 2.0f * PI * (float)rand()/RAND_MAX;
        rPhi_h[m] = cosf(phase);
        iPhi_h[m] = sinf(phase);
        rD_h[m] = (float)rand()/RAND_MAX - 0.5f;
        iD_h[m] = (float)rand()/RAND_MAX - 0.5f;
    }
    for (int n = 0; n < N; n++) {
        // 4x4x4 grid
        x_h[n] = (float)(n % 4);
        y_h[n] = (float)((n / 4) % 4);
        z_h[n] = (float)(n / 16);
    }

    // CPU reference
    printf("\nComputing CPU reference...\n");
    gpu_timer cpu_timer;
    cpu_timer.start();
    cpu_fhd(kx_h, ky_h, kz_h, x_h, y_h, z_h,
            rPhi_h, iPhi_h, rD_h, iD_h,
            rFhD_cpu, iFhD_cpu, M, N);
    cpu_timer.stop();
    printf("  CPU time: %.3f ms\n", cpu_timer.elapsed_ms());

    // Allocate device memory
    float *kx_d, *ky_d, *kz_d, *x_d, *y_d, *z_d;
    float *rPhi_d, *iPhi_d, *rD_d, *iD_d, *rMu_d, *iMu_d, *rFhD_d, *iFhD_d;
    CHECK_CUDA(cudaMalloc(&kx_d, szM)); CHECK_CUDA(cudaMalloc(&ky_d, szM));
    CHECK_CUDA(cudaMalloc(&kz_d, szM));
    CHECK_CUDA(cudaMalloc(&x_d, szN)); CHECK_CUDA(cudaMalloc(&y_d, szN));
    CHECK_CUDA(cudaMalloc(&z_d, szN));
    CHECK_CUDA(cudaMalloc(&rPhi_d, szM)); CHECK_CUDA(cudaMalloc(&iPhi_d, szM));
    CHECK_CUDA(cudaMalloc(&rD_d, szM)); CHECK_CUDA(cudaMalloc(&iD_d, szM));
    CHECK_CUDA(cudaMalloc(&rMu_d, szM)); CHECK_CUDA(cudaMalloc(&iMu_d, szM));
    CHECK_CUDA(cudaMalloc(&rFhD_d, szN)); CHECK_CUDA(cudaMalloc(&iFhD_d, szN));

    CHECK_CUDA(cudaMemcpy(kx_d, kx_h, szM, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ky_d, ky_h, szM, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(kz_d, kz_h, szM, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(x_d, x_h, szN, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(y_d, y_h, szN, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(z_d, z_h, szN, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(rPhi_d, rPhi_h, szM, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(iPhi_d, iPhi_h, szM, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(rD_d, rD_h, szM, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(iD_d, iD_h, szM, cudaMemcpyHostToDevice));

    int mu_blocks = (M + MU_THREADS_PER_BLOCK - 1) / MU_THREADS_PER_BLOCK;
    int fhd_blocks = (M + FHD_THREADS_PER_BLOCK - 1) / FHD_THREADS_PER_BLOCK;

    // Step 1: cmpMu kernel
    printf("\nLaunching cmpMu kernel: %d blocks x %d threads\n", mu_blocks, MU_THREADS_PER_BLOCK);
    cmpMu<<<mu_blocks, MU_THREADS_PER_BLOCK>>>(rPhi_d, iPhi_d, rD_d, iD_d, rMu_d, iMu_d, M);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // Step 2: cmpFhD scatter kernel (with atomics)
    printf("Launching cmpFhD scatter kernel: %d blocks x %d threads\n", fhd_blocks, FHD_THREADS_PER_BLOCK);

    // Zero output arrays
    CHECK_CUDA(cudaMemset(rFhD_d, 0, szN));
    CHECK_CUDA(cudaMemset(iFhD_d, 0, szN));

    // Warm-up
    cmpFhD_scatter<<<fhd_blocks, FHD_THREADS_PER_BLOCK>>>(kx_d, ky_d, kz_d,
        x_d, y_d, z_d, rMu_d, iMu_d, rFhD_d, iFhD_d, M, N);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Re-zero for timed run
    CHECK_CUDA(cudaMemset(rFhD_d, 0, szN));
    CHECK_CUDA(cudaMemset(iFhD_d, 0, szN));

    gpu_timer gpu_timer;
    gpu_timer.start();
    cmpFhD_scatter<<<fhd_blocks, FHD_THREADS_PER_BLOCK>>>(kx_d, ky_d, kz_d,
        x_d, y_d, z_d, rMu_d, iMu_d, rFhD_d, iFhD_d, M, N);
    gpu_timer.stop();
    float gpu_ms = gpu_timer.elapsed_ms();

    CHECK_CUDA(cudaMemcpy(rFhD_h, rFhD_d, szN, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(iFhD_h, iFhD_d, szN, cudaMemcpyDeviceToHost));

    printf("  GPU cmpFhD time: %.3f ms\n", gpu_ms);

    // Validate
    printf("\nValidating against CPU reference...\n");
    float tol = 1e-4f;
    bool pass_r = cpu_allclose(rFhD_h, rFhD_cpu, N, tol);
    bool pass_i = cpu_allclose(iFhD_h, iFhD_cpu, N, tol);

    if (pass_r && pass_i) {
        printf("  Validation: PASS (both real and imaginary components)\n");
    } else {
        if (!pass_r) printf("  FAIL: Real component mismatch\n");
        if (!pass_i) printf("  FAIL: Imaginary component mismatch\n");
        for (int i = 0; i < 5 && i < N; i++) {
            printf("  [%d] real: gpu=%.6f cpu=%.6f  imag: gpu=%.6f cpu=%.6f\n",
                   i, rFhD_h[i], rFhD_cpu[i], iFhD_h[i], iFhD_cpu[i]);
        }
    }

    // Cleanup
    cudaFree(kx_d); cudaFree(ky_d); cudaFree(kz_d);
    cudaFree(x_d); cudaFree(y_d); cudaFree(z_d);
    cudaFree(rPhi_d); cudaFree(iPhi_d); cudaFree(rD_d); cudaFree(iD_d);
    cudaFree(rMu_d); cudaFree(iMu_d); cudaFree(rFhD_d); cudaFree(iFhD_d);
    free(kx_h); free(ky_h); free(kz_h); free(x_h); free(y_h); free(z_h);
    free(rPhi_h); free(iPhi_h); free(rD_h); free(iD_h);
    free(rFhD_h); free(iFhD_h); free(rFhD_cpu); free(iFhD_cpu);

    printf("\nDone.\n");
    return (pass_r && pass_i) ? 0 : 1;
}
