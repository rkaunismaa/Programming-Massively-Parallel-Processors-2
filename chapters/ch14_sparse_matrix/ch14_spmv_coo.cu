/*
 * =============================================================================
 *  ch14_spmv_coo.cu — Parallel Sparse Matrix-Vector Multiplication
 *                     using the COO (Coordinate List) format
 *
 *  Book:      Programming Massively Parallel Processors (4th Ed.)
 *  Figure:    14.5 — SpMV/COO kernel
 *
 *  Summary:
 *    Each thread is assigned one nonzero element. The thread reads
 *    value[i], colIdx[i], and rowIdx[i] from the COO arrays, then
 *    atomically accumulates value[i] * x[colIdx[i]] into y[rowIdx[i]].
 *
 *  Warp-level atomicAdd is efficient for this pattern because multiple
 *    threads may target the same output row.
 *
 *  Hardware:  GTX 1050 (sm_61, device 1)
 * =============================================================================
 */

#include "../common/cuda_utils.cuh"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>

// ---------------------------------------------------------------------------
//  SpMV/COO kernel (Fig 14.5)
// ---------------------------------------------------------------------------
//  One thread per nonzero element. AtomicAdd handles row collisions.
// ---------------------------------------------------------------------------
__global__ void spmv_coo_kernel(const float *value, const int *colIdx,
                                const int *rowIdx, const float *x,
                                float *y, int numNonzeros) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numNonzeros) {
        int row = rowIdx[i];
        int col = colIdx[i];
        float val = value[i];
        atomicAdd(&y[row], val * x[col]);
    }
}

// ---------------------------------------------------------------------------
//  CPU reference: dense SpMV (for validation)
// ---------------------------------------------------------------------------
void host_spmv_dense(const float *A, const float *x, float *y,
                     int rows, int cols) {
    for (int r = 0; r < rows; r++) {
        float sum = 0.0f;
        for (int c = 0; c < cols; c++) {
            sum += A[r * cols + c] * x[c];
        }
        y[r] = sum;
    }
}

// ---------------------------------------------------------------------------
//  CPU reference: SpMV from COO arrays
// ---------------------------------------------------------------------------
void host_spmv_coo(const float *value, const int *colIdx, const int *rowIdx,
                   const float *x, float *y, int numNonzeros) {
    // Note: y is NOT zeroed — caller must memset(y, 0, ...) first
    for (int i = 0; i < numNonzeros; i++) {
        int row = rowIdx[i];
        int col = colIdx[i];
        y[row] += value[i] * x[col];
    }
}

// ---------------------------------------------------------------------------
//  Main
// ---------------------------------------------------------------------------
int main() {
    // ---- select GTX 1050 (device 1) ----
    CHECK_CUDA(cudaSetDevice(1));
    print_device_info(1);

    printf("╔══════════════════════════════════════════════════════════╗\n");
    printf("║  SpMV/COO — Sparse Matrix-Vector Multiplication          ║\n");
    printf("║  COO (Coordinate List) format                            ║\n");
    printf("╚══════════════════════════════════════════════════════════╝\n\n");

    // -----------------------------------------------------------------------
    //  Test 1: Book example — 4×4 matrix from Fig 14.1
    //     [1 0 7 0]     Nonzeros: value=[1,7,8,4,3,2,1]
    //     [0 0 8 0]               colIdx=[0,2,2,1,2,0,3]
    //     [0 4 3 0]               rowIdx=[0,0,1,2,2,3,3]
    //     [2 0 0 1]
    // -----------------------------------------------------------------------
    printf("--- Test 1: Book example 4×4 matrix ---\n");

    const int rows1 = 4, cols1 = 4;
    const int nnz1 = 7;

    float h_dense1[16] = {
        1, 0, 7, 0,
        0, 0, 8, 0,
        0, 4, 3, 0,
        2, 0, 0, 1
    };
    float h_x1[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float h_y_ref1[4] = {0};

    float h_value1[7]   = {1, 7, 8, 4, 3, 2, 1};
    int   h_colIdx1[7]  = {0, 2, 2, 1, 2, 0, 3};
    int   h_rowIdx1[7]  = {0, 0, 1, 2, 2, 3, 3};
    float h_y_gpu1[4]   = {0};

    // CPU reference (dense)
    host_spmv_dense(h_dense1, h_x1, h_y_ref1, rows1, cols1);

    // Device allocations
    float *d_value1, *d_x1, *d_y1;
    int   *d_colIdx1, *d_rowIdx1;
    CHECK_CUDA(cudaMalloc(&d_value1,  nnz1 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_colIdx1, nnz1 * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_rowIdx1, nnz1 * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_x1,      cols1 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_y1,      rows1 * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_value1,  h_value1,  nnz1 * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_colIdx1, h_colIdx1, nnz1 * sizeof(int),   cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_rowIdx1, h_rowIdx1, nnz1 * sizeof(int),   cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_x1,      h_x1,      cols1 * sizeof(float),cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_y1,      0,         rows1 * sizeof(float)));

    int blockSize = 256;
    int gridSize  = (nnz1 + blockSize - 1) / blockSize;

    spmv_coo_kernel<<<gridSize, blockSize>>>(d_value1, d_colIdx1, d_rowIdx1,
                                             d_x1, d_y1, nnz1);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_y_gpu1, d_y1, rows1 * sizeof(float), cudaMemcpyDeviceToHost));

    printf("  Expected y: [%.2f, %.2f, %.2f, %.2f]\n",
           h_y_ref1[0], h_y_ref1[1], h_y_ref1[2], h_y_ref1[3]);
    printf("  Got y:      [%.2f, %.2f, %.2f, %.2f]\n",
           h_y_gpu1[0], h_y_gpu1[1], h_y_gpu1[2], h_y_gpu1[3]);

    bool pass1 = cpu_allclose(h_y_ref1, h_y_gpu1, rows1);
    printf("  %s\n\n", pass1 ? "VALIDATION: PASS" : "VALIDATION: FAIL");

    CHECK_CUDA(cudaFree(d_value1));
    CHECK_CUDA(cudaFree(d_colIdx1));
    CHECK_CUDA(cudaFree(d_rowIdx1));
    CHECK_CUDA(cudaFree(d_x1));
    CHECK_CUDA(cudaFree(d_y1));

    // -----------------------------------------------------------------------
    //  Test 2: Larger random sparse matrix
    // -----------------------------------------------------------------------
    printf("--- Test 2: Random 1024×1024 sparse matrix (1%% density) ---\n");

    const int rows2 = 1024, cols2 = 1024;
    const float density = 0.01f;  // 1% nonzeros
    const int max_nnz = (int)(rows2 * cols2 * density) + rows2;  // at least 1 per row

    // Host arrays for COO
    float *h_value2  = new float[max_nnz];
    int   *h_colIdx2 = new int[max_nnz];
    int   *h_rowIdx2 = new int[max_nnz];
    float *h_x2      = new float[cols2];
    float *h_y_ref2  = new float[rows2]();
    float *h_y_gpu2  = new float[rows2]();

    // Seed for reproducibility
    srand(42);

    int actual_nnz = 0;
    for (int r = 0; r < rows2; r++) {
        for (int c = 0; c < cols2; c++) {
            if ((rand() % 100) < (int)(density * 100)) {
                h_value2[actual_nnz]  = (float)(rand() % 10) / 3.0f + 0.5f;
                h_colIdx2[actual_nnz] = c;
                h_rowIdx2[actual_nnz] = r;
                actual_nnz++;
            }
        }
        // Ensure at least 1 nonzero per row
        if (actual_nnz > 0 && h_rowIdx2[actual_nnz - 1] < r) {
            int c = rand() % cols2;
            h_value2[actual_nnz]  = (float)(rand() % 10) / 3.0f + 0.5f;
            h_colIdx2[actual_nnz] = c;
            h_rowIdx2[actual_nnz] = r;
            actual_nnz++;
        }
    }

    printf("  Nonzeros: %d\n", actual_nnz);

    for (int i = 0; i < cols2; i++)
        h_x2[i] = (float)(rand() % 100) / 20.0f;

    // CPU reference
    host_spmv_coo(h_value2, h_colIdx2, h_rowIdx2, h_x2, h_y_ref2, actual_nnz);

    // Device
    float *d_value2, *d_x2, *d_y2;
    int   *d_colIdx2, *d_rowIdx2;
    CHECK_CUDA(cudaMalloc(&d_value2,  actual_nnz * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_colIdx2, actual_nnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_rowIdx2, actual_nnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_x2,      cols2 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_y2,      rows2 * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_value2,  h_value2,  actual_nnz * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_colIdx2, h_colIdx2, actual_nnz * sizeof(int),   cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_rowIdx2, h_rowIdx2, actual_nnz * sizeof(int),   cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_x2,      h_x2,      cols2 * sizeof(float),      cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_y2,      0,         rows2 * sizeof(float)));

    gridSize = (actual_nnz + blockSize - 1) / blockSize;

    gpu_timer timer;
    // Warm-up
    spmv_coo_kernel<<<gridSize, blockSize>>>(d_value2, d_colIdx2, d_rowIdx2,
                                             d_x2, d_y2, actual_nnz);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Restore zeroed y
    CHECK_CUDA(cudaMemset(d_y2, 0, rows2 * sizeof(float)));

    // Timed run
    timer.start();
    spmv_coo_kernel<<<gridSize, blockSize>>>(d_value2, d_colIdx2, d_rowIdx2,
                                             d_x2, d_y2, actual_nnz);
    timer.stop();
    float ms2 = timer.elapsed_ms();

    CHECK_CUDA(cudaMemcpy(h_y_gpu2, d_y2, rows2 * sizeof(float), cudaMemcpyDeviceToHost));

    bool pass2 = cpu_allclose(h_y_ref2, h_y_gpu2, rows2);
    printf("  Kernel time: %.3f ms\n", ms2);
    printf("  Throughput:  %.2f M nonzeros/sec\n",
           actual_nnz / ms2 / 1000.0f);
    printf("  %s\n\n", pass2 ? "VALIDATION: PASS" : "VALIDATION: FAIL");

    CHECK_CUDA(cudaFree(d_value2));
    CHECK_CUDA(cudaFree(d_colIdx2));
    CHECK_CUDA(cudaFree(d_rowIdx2));
    CHECK_CUDA(cudaFree(d_x2));
    CHECK_CUDA(cudaFree(d_y2));

    delete[] h_value2;
    delete[] h_colIdx2;
    delete[] h_rowIdx2;
    delete[] h_x2;
    delete[] h_y_ref2;
    delete[] h_y_gpu2;

    printf("Done.\n");
    return (pass1 && pass2) ? 0 : 1;
}
