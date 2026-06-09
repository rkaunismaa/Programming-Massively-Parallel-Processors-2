// Book:      Programming Massively Parallel Processors (Kirk, Hwu & El Hajj, 4th ed.)
// Chapter:   3 — Multidimensional Grids and Data
// Exercise:  3.2 — Matrix-vector multiplication
// Concept:   A = B * C where B is a square matrix and C is a vector
//            Each thread computes one element of output vector A
//            A[i] = sum over j of B[i][j] * C[j]
// Key insight: Same 2D thread mapping as matrix-matrix multiply, but output
//              is a 1D vector; each thread computes dot product of one matrix row
//              with the entire vector
// Compile:   nvcc -std=c++17 -arch=sm_89 -O2 ex02_matvec.cu -o ex02_matvec

#include "../../common/cuda_utils.cuh"
#include <cstring>

// ============================================================
// Kernel: matrix-vector multiplication A = B * C
// Each thread computes one element of output vector A
// ============================================================
__global__
void matVecKernel(const float* B, const float* C, float* A, int Width) {
    // Each thread handles one element of the output vector
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < Width) {
        float Avalue = 0.0f;
        // Dot product of row 'idx' of B with vector C
        for (int j = 0; j < Width; j++) {
            Avalue += B[idx * Width + j] * C[j];
        }
        A[idx] = Avalue;
    }
}

// ============================================================
// Host wrapper
// ============================================================
void matrixVectorMul(const float* B_h, const float* C_h, float* A_h, int Width) {
    int matrix_size = Width * Width * sizeof(float);
    int vector_size = Width * sizeof(float);

    float *B_d, *C_d, *A_d;
    CHECK_CUDA(cudaMalloc((void**)&B_d, matrix_size));
    CHECK_CUDA(cudaMalloc((void**)&C_d, vector_size));
    CHECK_CUDA(cudaMalloc((void**)&A_d, vector_size));

    CHECK_CUDA(cudaMemcpy(B_d, B_h, matrix_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(C_d, C_h, vector_size, cudaMemcpyHostToDevice));

    // 1D grid — one thread per output vector element
    int threads_per_block = 256;
    int blocks = (Width + threads_per_block - 1) / threads_per_block;

    matVecKernel<<<blocks, threads_per_block>>>(B_d, C_d, A_d, Width);
    CHECK_CUDA(cudaGetLastError());

    CHECK_CUDA(cudaMemcpy(A_h, A_d, vector_size, cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(B_d));
    CHECK_CUDA(cudaFree(C_d));
    CHECK_CUDA(cudaFree(A_d));
}

// ============================================================
// CPU reference
// ============================================================
void cpuMatrixVectorMul(const float* B, const float* C, float* A, int Width) {
    for (int i = 0; i < Width; i++) {
        float sum = 0.0f;
        for (int j = 0; j < Width; j++) {
            sum += B[i * Width + j] * C[j];
        }
        A[i] = sum;
    }
}

int main() {
    print_device_info();

    int Width = 4096;

    printf("\nExercise 3.2: Matrix-vector multiplication\n");
    printf("Matrix size: %d x %d\n", Width, Width);
    printf("Vector size: %d\n", Width);

    float* B_h = new float[Width * Width];
    float* C_h = new float[Width];
    float* A_h = new float[Width];
    float* expected = new float[Width];

    for (int i = 0; i < Width * Width; i++) {
        B_h[i] = (float)((i * 7 + 3) % 100) / 100.0f;
    }
    for (int i = 0; i < Width; i++) {
        C_h[i] = (float)((i * 13 + 7) % 100) / 100.0f;
    }

    cpuMatrixVectorMul(B_h, C_h, expected, Width);

    gpu_timer timer;
    timer.start();
    matrixVectorMul(B_h, C_h, A_h, Width);
    timer.stop();
    float gpu_ms = timer.elapsed_ms();

    bool pass = cpu_allclose(expected, A_h, Width, 1e-2f);

    printf("\nValidation: %s\n", pass ? "PASS" : "FAIL");
    printf("GPU time:   %.3f ms\n", gpu_ms);
    // GFLOPS: 2 * Width^2 FLOPs (multiply + add per inner loop iteration)
    printf("GFLOPS:     %.2f\n", 2.0 * Width * Width / (gpu_ms * 1e6));

    delete[] B_h;
    delete[] C_h;
    delete[] A_h;
    delete[] expected;

    return pass ? 0 : 1;
}
