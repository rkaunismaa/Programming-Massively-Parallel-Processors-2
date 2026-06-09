// Book:      Programming Massively Parallel Processors (Kirk, Hwu & El Hajj, 4th ed.)
// Chapter:   3 — Multidimensional Grids and Data
// Exercise:  3.1a — Matrix multiplication: one thread per output row
// Concept:   Each thread computes an entire row of the output matrix P
//            Grid: Width blocks in y-dimension, 1 block in x-dimension
//            Block: blockDim.x threads (one thread handles one row's computation)
// Key insight: Thread's row index determines which row of P to compute;
//              inner loop over columns and k (dot product dimension)
// Compile:   nvcc -std=c++17 -arch=sm_89 -O2 ex01a.cu -o ex01a

#include "../../common/cuda_utils.cuh"
#include <cstring>

// ============================================================
// Kernel: each thread computes one full row of P = M * N
// ============================================================
__global__
void matMulOneRowKernel(const float* M, const float* N, float* P, int Width) {
    // Each thread handles one row of the output matrix
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < Width) {
        // Compute all columns for this row
        for (int col = 0; col < Width; col++) {
            float Pvalue = 0.0f;
            // Dot product of row 'row' of M and column 'col' of N
            for (int k = 0; k < Width; k++) {
                Pvalue += M[row * Width + k] * N[k * Width + col];
            }
            P[row * Width + col] = Pvalue;
        }
    }
}

// ============================================================
// Host wrapper
// ============================================================
void matrixMulOneRow(const float* M_h, const float* N_h, float* P_h, int Width) {
    int size = Width * Width * sizeof(float);

    float *M_d, *N_d, *P_d;
    CHECK_CUDA(cudaMalloc((void**)&M_d, size));
    CHECK_CUDA(cudaMalloc((void**)&N_d, size));
    CHECK_CUDA(cudaMalloc((void**)&P_d, size));

    CHECK_CUDA(cudaMemcpy(M_d, M_h, size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(N_d, N_h, size, cudaMemcpyHostToDevice));

    // Grid: one block per row of output matrix
    dim3 block(1, 256);  // 256 threads in y-dimension
    dim3 grid(1, (Width + block.y - 1) / block.y);

    matMulOneRowKernel<<<grid, block>>>(M_d, N_d, P_d, Width);
    CHECK_CUDA(cudaGetLastError());

    CHECK_CUDA(cudaMemcpy(P_h, P_d, size, cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(M_d));
    CHECK_CUDA(cudaFree(N_d));
    CHECK_CUDA(cudaFree(P_d));
}

// ============================================================
// CPU reference
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

    int Width = 1024;

    printf("\nExercise 3.1a: One thread per output row\n");
    printf("Matrix size: %d x %d\n", Width, Width);

    float* M_h = new float[Width * Width];
    float* N_h = new float[Width * Width];
    float* P_h = new float[Width * Width];
    float* expected = new float[Width * Width];

    for (int i = 0; i < Width * Width; i++) {
        M_h[i] = (float)((i * 7 + 3) % 100) / 100.0f;
        N_h[i] = (float)((i * 13 + 7) % 100) / 100.0f;
    }

    cpuMatrixMul(M_h, N_h, expected, Width);

    gpu_timer timer;
    timer.start();
    matrixMulOneRow(M_h, N_h, P_h, Width);
    timer.stop();
    float gpu_ms = timer.elapsed_ms();

    bool pass = cpu_allclose_matrix(expected, P_h, Width, Width, 1e-2f);

    printf("\nValidation: %s\n", pass ? "PASS" : "FAIL");
    printf("GPU time:   %.3f ms\n", gpu_ms);
    printf("GFLOPS:     %.2f\n", 2.0 * Width * Width * Width / (gpu_ms * 1e6));

    delete[] M_h;
    delete[] N_h;
    delete[] P_h;
    delete[] expected;

    return pass ? 0 : 1;
}
