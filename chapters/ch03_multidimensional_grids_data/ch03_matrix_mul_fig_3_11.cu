// Book:      Programming Massively Parallel Processors (Kirk, Hwu & El Hajj, 4th ed.)
// Chapter:   3 — Multidimensional Grids and Data
// Reference: Figure 3.11
// Concept:   Matrix multiplication kernel — each thread computes one output element
// Key insight: P[row][col] = dot product of M's row and N's column; row-major
//              linearization for both M and N; square matrices only
// Hardware:  RTX 4090, sm_89 (Ada Lovelace)
// Compile:   nvcc -std=c++17 -arch=sm_89 -O2 ch03_matrix_mul_fig_3_11.cu -o matrix_mul

#include "../common/cuda_utils.cuh"
#include <cstring>

// ============================================================
// Kernel: matrixMulKernel (Figure 3.11)
// Each thread computes one element of output matrix P = M * N
// Square matrices of Width x Width, stored in row-major order
// ============================================================
#define BLOCK_WIDTH 16

__global__
void matrixMulKernel(const float* M, const float* N, float* P, int Width) {
    // Compute row and column indices for the output element
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Boundary check
    if (row < Width && col < Width) {
        float Pvalue = 0.0f;

        // Dot product of row 'row' of M and column 'col' of N
        for (int k = 0; k < Width; k++) {
            Pvalue += M[row * Width + k] * N[k * Width + col];
        }

        // Write result to output matrix P
        P[row * Width + col] = Pvalue;
    }
}

// ============================================================
// Host wrapper
// ============================================================
void matrixMul(const float* M_h, const float* N_h, float* P_h, int Width) {
    int size = Width * Width * sizeof(float);

    float *M_d, *N_d, *P_d;
    CHECK_CUDA(cudaMalloc((void**)&M_d, size));
    CHECK_CUDA(cudaMalloc((void**)&N_d, size));
    CHECK_CUDA(cudaMalloc((void**)&P_d, size));

    CHECK_CUDA(cudaMemcpy(M_d, M_h, size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(N_d, N_h, size, cudaMemcpyHostToDevice));

    // 2D block and grid configuration
    dim3 block(BLOCK_WIDTH, BLOCK_WIDTH);
    dim3 grid((Width + block.x - 1) / block.x,
              (Width + block.y - 1) / block.y);

    matrixMulKernel<<<grid, block>>>(M_d, N_d, P_d, Width);
    CHECK_CUDA(cudaGetLastError());

    CHECK_CUDA(cudaMemcpy(P_h, P_d, size, cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(M_d));
    CHECK_CUDA(cudaFree(N_d));
    CHECK_CUDA(cudaFree(P_d));
}

// ============================================================
// CPU reference for matrix multiplication
// ============================================================
void cpuMatrixMul(const float* M, const float* N, float* P, int Width) {
    for (int i = 0; i < Width; i++) {
        for (int j = 0; j < Width; j++) {
            float sum = 0.0f;
            for (int k = 0; k < Width; k++) {
                sum += M[i * Width + k] * N[k * Width + j];
            }
            P[i * Width + j] = sum;
        }
    }
}

int main() {
    print_device_info();

    // Use a moderate matrix size — large enough to show GPU benefit,
    // small enough to keep CPU reference tractable
    int Width = 1024;

    printf("\nMatrix size: %d x %d\n", Width, Width);
    printf("Block size:  %d x %d\n", BLOCK_WIDTH, BLOCK_WIDTH);

    // Allocate host memory
    float* M_h = new float[Width * Width];
    float* N_h = new float[Width * Width];
    float* P_h = new float[Width * Width];
    float* expected = new float[Width * Width];

    // Initialize with small deterministic values to avoid overflow
    for (int i = 0; i < Width * Width; i++) {
        M_h[i] = (float)((i * 7 + 3) % 100) / 100.0f;
        N_h[i] = (float)((i * 13 + 7) % 100) / 100.0f;
    }

    // CPU reference
    cpuMatrixMul(M_h, N_h, expected, Width);

    // GPU execution with timing
    gpu_timer timer;
    timer.start();
    matrixMul(M_h, N_h, P_h, Width);
    timer.stop();
    float gpu_ms = timer.elapsed_ms();

    // Validate
    bool pass = cpu_allclose_matrix(expected, P_h, Width, Width, 1e-2f);

    printf("\nValidation: %s\n", pass ? "PASS" : "FAIL");
    printf("GPU time:   %.3f ms\n", gpu_ms);

    // GFLOPS: 2 * Width^3 FLOPs (multiply + add per inner loop iteration)
    double gflops = 2.0 * Width * Width * Width / (gpu_ms * 1e6);
    printf("GFLOPS:     %.2f\n", gflops);

    delete[] M_h;
    delete[] N_h;
    delete[] P_h;
    delete[] expected;

    return pass ? 0 : 1;
}
