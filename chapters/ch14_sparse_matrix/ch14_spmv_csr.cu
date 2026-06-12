/*
 * =============================================================================
 *  ch14_spmv_csr.cu — Parallel Sparse Matrix-Vector Multiplication
 *                     using the CSR (Compressed Sparse Row) format
 *
 *  Book:      Programming Massively Parallel Processors (4th Ed.)
 *  Figure:    14.9 — SpMV/CSR kernel
 *
 *  Summary:
 *    One thread per row. Each thread reads rowPtrs[row] to find its
 *    starting index into colIdx[] and value[], and rowPtrs[row+1] for
 *    the end. The thread accumulates value[i] * x[colIdx[i]] in a local
 *    register sum, then writes y[row] = sum — no atomics needed since
 *    each thread owns a unique row.
 *
 *  Drawbacks (per textbook):
 *    - Non-coalesced global memory access (adjacent rows' nonzeros
 *      are not adjacent in memory)
 *    - High control divergence (rows have varying numbers of nonzeros)
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
//  SpMV/CSR kernel (Fig 14.9)
// ---------------------------------------------------------------------------
//  One thread per row. No atomics — each thread owns a distinct output row.
// ---------------------------------------------------------------------------
__global__ void spmv_csr_kernel(const float *value, const int *colIdx,
                                const int *rowPtrs, const float *x,
                                float *y, int numRows) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < numRows) {
        float sum = 0.0f;
        int start = rowPtrs[row];
        int end   = rowPtrs[row + 1];
        for (int i = start; i < end; i++) {
            sum += value[i] * x[colIdx[i]];
        }
        y[row] = sum;
    }
}

// ---------------------------------------------------------------------------
//  CPU reference: SpMV from CSR arrays
// ---------------------------------------------------------------------------
void host_spmv_csr(const float *value, const int *colIdx, const int *rowPtrs,
                   const float *x, float *y, int numRows) {
    for (int row = 0; row < numRows; row++) {
        float sum = 0.0f;
        for (int i = rowPtrs[row]; i < rowPtrs[row + 1]; i++) {
            sum += value[i] * x[colIdx[i]];
        }
        y[row] = sum;
    }
}

// ---------------------------------------------------------------------------
//  Helper: build CSR arrays from a dense matrix (simple test generation)
// ---------------------------------------------------------------------------
void dense_to_csr(const float *dense, int rows, int cols,
                  float *&value, int *&colIdx, int *&rowPtrs,
                  int &nnz) {
    // Count nonzeros
    nnz = 0;
    for (int r = 0; r < rows; r++)
        for (int c = 0; c < cols; c++)
            if (std::fabs(dense[r * cols + c]) > 1e-6f)
                nnz++;

    value  = new float[nnz];
    colIdx = new int[nnz];
    rowPtrs = new int[rows + 1];

    int idx = 0;
    for (int r = 0; r < rows; r++) {
        rowPtrs[r] = idx;
        for (int c = 0; c < cols; c++) {
            float v = dense[r * cols + c];
            if (std::fabs(v) > 1e-6f) {
                value[idx]  = v;
                colIdx[idx] = c;
                idx++;
            }
        }
    }
    rowPtrs[rows] = idx;
}

// ---------------------------------------------------------------------------
//  Main
// ---------------------------------------------------------------------------
int main() {
    CHECK_CUDA(cudaSetDevice(1));
    print_device_info(1);

    printf("╔══════════════════════════════════════════════════════════╗\n");
    printf("║  SpMV/CSR — Sparse Matrix-Vector Multiplication          ║\n");
    printf("║  CSR (Compressed Sparse Row) format                     ║\n");
    printf("╚══════════════════════════════════════════════════════════╝\n\n");

    // -----------------------------------------------------------------------
    //  Test 1: Book example — 4×4 matrix from Fig 14.1
    // -----------------------------------------------------------------------
    printf("--- Test 1: Book example 4×4 matrix ---\n");

    const int rows1 = 4, cols1 = 4;
    float dense1[16] = {
        1, 0, 7, 0,
        0, 0, 8, 0,
        0, 4, 3, 0,
        2, 0, 0, 1
    };
    float h_x1[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float h_y_ref1[4] = {0};

    float *h_value1 = nullptr;
    int   *h_colIdx1 = nullptr, *h_rowPtrs1 = nullptr;
    int    nnz1 = 0;
    dense_to_csr(dense1, rows1, cols1, h_value1, h_colIdx1, h_rowPtrs1, nnz1);
    float h_y_gpu1[4] = {0};

    printf("  Nonzeros: %d\n", nnz1);
    printf("  Row pointers: [%d, %d, %d, %d, %d]\n",
           h_rowPtrs1[0], h_rowPtrs1[1], h_rowPtrs1[2],
           h_rowPtrs1[3], h_rowPtrs1[4]);

    // CPU reference (dense)
    for (int r = 0; r < rows1; r++)
        for (int c = 0; c < cols1; c++)
            h_y_ref1[r] += dense1[r * cols1 + c] * h_x1[c];

    // Device allocations
    float *d_value1, *d_x1, *d_y1;
    int   *d_colIdx1, *d_rowPtrs1;
    CHECK_CUDA(cudaMalloc(&d_value1,   nnz1 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_colIdx1,  nnz1 * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_rowPtrs1, (rows1 + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_x1,       cols1 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_y1,       rows1 * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_value1,   h_value1,   nnz1 * sizeof(float),
                           cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_colIdx1,  h_colIdx1,  nnz1 * sizeof(int),
                           cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_rowPtrs1, h_rowPtrs1, (rows1 + 1) * sizeof(int),
                           cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_x1,       h_x1,       cols1 * sizeof(float),
                           cudaMemcpyHostToDevice));

    int blockSize = 256;
    int gridSize  = (rows1 + blockSize - 1) / blockSize;

    spmv_csr_kernel<<<gridSize, blockSize>>>(d_value1, d_colIdx1, d_rowPtrs1,
                                             d_x1, d_y1, rows1);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_y_gpu1, d_y1, rows1 * sizeof(float),
                           cudaMemcpyDeviceToHost));

    printf("  Expected y: [%.2f, %.2f, %.2f, %.2f]\n",
           h_y_ref1[0], h_y_ref1[1], h_y_ref1[2], h_y_ref1[3]);
    printf("  Got y:      [%.2f, %.2f, %.2f, %.2f]\n",
           h_y_gpu1[0], h_y_gpu1[1], h_y_gpu1[2], h_y_gpu1[3]);

    bool pass1 = cpu_allclose(h_y_ref1, h_y_gpu1, rows1);
    printf("  %s\n\n", pass1 ? "VALIDATION: PASS" : "VALIDATION: FAIL");

    delete[] h_value1;
    delete[] h_colIdx1;
    delete[] h_rowPtrs1;
    CHECK_CUDA(cudaFree(d_value1));
    CHECK_CUDA(cudaFree(d_colIdx1));
    CHECK_CUDA(cudaFree(d_rowPtrs1));
    CHECK_CUDA(cudaFree(d_x1));
    CHECK_CUDA(cudaFree(d_y1));

    // -----------------------------------------------------------------------
    //  Test 2: Random 4096×4096 sparse matrix (~0.5% density)
    // -----------------------------------------------------------------------
    printf("--- Test 2: Random 4096×4096 sparse matrix (~0.5%% density) ---\n");

    const int rows2 = 4096, cols2 = 4096;
    const float density = 0.005f;
    const int max_nnz2 = (int)(rows2 * cols2 * density) + rows2 * 2;

    float *h_value2  = new float[max_nnz2];
    int   *h_colIdx2  = new int[max_nnz2];
    int   *h_rowPtrs2 = new int[rows2 + 1];
    float *h_x2       = new float[cols2];
    float *h_y_ref2   = new float[rows2]();
    float *h_y_gpu2   = new float[rows2]();

    srand(123);

    int nnz2 = 0;
    for (int r = 0; r < rows2; r++) {
        h_rowPtrs2[r] = nnz2;
        int row_nnz = 0;
        for (int c = 0; c < cols2; c++) {
            if ((rand() % 1000) < (int)(density * 1000)) {
                h_value2[nnz2]  = (float)(rand() % 20) / 5.0f + 0.1f;
                h_colIdx2[nnz2] = c;
                nnz2++;
                row_nnz++;
            }
        }
        // At least 1 nonzero per row
        if (row_nnz == 0) {
            int c = rand() % cols2;
            h_value2[nnz2]  = 1.0f;
            h_colIdx2[nnz2] = c;
            nnz2++;
        }
    }
    h_rowPtrs2[rows2] = nnz2;

    printf("  Nonzeros: %d\n", nnz2);

    for (int i = 0; i < cols2; i++)
        h_x2[i] = (float)(rand() % 100) / 25.0f;

    // CPU reference
    host_spmv_csr(h_value2, h_colIdx2, h_rowPtrs2, h_x2, h_y_ref2, rows2);

    // Device
    float *d_value2, *d_x2, *d_y2;
    int   *d_colIdx2, *d_rowPtrs2;
    CHECK_CUDA(cudaMalloc(&d_value2,   nnz2 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_colIdx2,  nnz2 * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_rowPtrs2, (rows2 + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_x2,       cols2 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_y2,       rows2 * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_value2,   h_value2,   nnz2 * sizeof(float),
                           cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_colIdx2,  h_colIdx2,  nnz2 * sizeof(int),
                           cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_rowPtrs2, h_rowPtrs2, (rows2 + 1) * sizeof(int),
                           cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_x2,       h_x2,       cols2 * sizeof(float),
                           cudaMemcpyHostToDevice));

    gridSize = (rows2 + blockSize - 1) / blockSize;

    // Warm-up
    spmv_csr_kernel<<<gridSize, blockSize>>>(d_value2, d_colIdx2, d_rowPtrs2,
                                             d_x2, d_y2, rows2);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemset(d_y2, 0, rows2 * sizeof(float)));

    // Timed run
    gpu_timer timer;
    timer.start();
    spmv_csr_kernel<<<gridSize, blockSize>>>(d_value2, d_colIdx2, d_rowPtrs2,
                                             d_x2, d_y2, rows2);
    timer.stop();
    float ms2 = timer.elapsed_ms();

    CHECK_CUDA(cudaMemcpy(h_y_gpu2, d_y2, rows2 * sizeof(float),
                           cudaMemcpyDeviceToHost));

    bool pass2 = cpu_allclose(h_y_ref2, h_y_gpu2, rows2);
    printf("  Kernel time: %.3f ms\n", ms2);
    printf("  Throughput:  %.2f M nonzeros/sec\n", nnz2 / ms2 / 1000.0f);
    printf("  %s\n\n", pass2 ? "VALIDATION: PASS" : "VALIDATION: FAIL");

    delete[] h_value2;
    delete[] h_colIdx2;
    delete[] h_rowPtrs2;
    delete[] h_x2;
    delete[] h_y_ref2;
    delete[] h_y_gpu2;
    CHECK_CUDA(cudaFree(d_value2));
    CHECK_CUDA(cudaFree(d_colIdx2));
    CHECK_CUDA(cudaFree(d_rowPtrs2));
    CHECK_CUDA(cudaFree(d_x2));
    CHECK_CUDA(cudaFree(d_y2));

    printf("Done.\n");
    return (pass1 && pass2) ? 0 : 1;
}
