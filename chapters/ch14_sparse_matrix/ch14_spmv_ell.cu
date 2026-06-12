/*
 * =============================================================================
 *  ch14_spmv_ell.cu — Parallel Sparse Matrix-Vector Multiplication
 *                     using the ELL (ELLPACK) format
 *
 *  Book:      Programming Massively Parallel Processors (4th Ed.)
 *  Figure:    14.12 — SpMV/ELL kernel
 *
 *  Summary:
 *    The ELL format pads all rows to the same length (max nonzeros per row),
 *    then stores the data in column-major order. This transposition enables
 *    fully coalesced memory accesses: consecutive threads (consecutive rows)
 *    access consecutive memory locations in the value[] and colIdx[] arrays.
 *
 *    Index formula:  i = t * numRows + row
 *    where t is the iteration index and row is the row assigned to the thread.
 *
 *    This kernel assumes padding elements have value 0, which does not affect
 *    the result. An optional nnzPerRow array can be used to limit iterations.
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
//  SpMV/ELL kernel (Fig 14.12)
// ---------------------------------------------------------------------------
//  One thread per row. Column-major access → coalesced.
//  The simpler version iterates all numColsPerRow (padding = 0 has no effect).
// ---------------------------------------------------------------------------
__global__ void spmv_ell_kernel(const float *value, const int *colIdx,
                                int numRows, int numColsPerRow,
                                const float *x, float *y) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < numRows) {
        float sum = 0.0f;
        for (int t = 0; t < numColsPerRow; t++) {
            // Column-major index: t-th element of every row
            int idx = t * numRows + row;
            int col = colIdx[idx];
            float val = value[idx];
            sum += val * x[col];
        }
        y[row] = sum;
    }
}

// ---------------------------------------------------------------------------
//  Alternative: ELL kernel with nnzPerRow (fewer iterations for short rows)
// ---------------------------------------------------------------------------
__global__ void spmv_ell_nnz_kernel(const float *value, const int *colIdx,
                                    const int *nnzPerRow,
                                    int numRows, int numColsPerRow,
                                    const float *x, float *y) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < numRows) {
        float sum = 0.0f;
        int iters = nnzPerRow[row];  // only iterate actual nonzeros
        for (int t = 0; t < iters; t++) {
            int idx = t * numRows + row;
            int col = colIdx[idx];
            float val = value[idx];
            sum += val * x[col];
        }
        y[row] = sum;
    }
}

// ---------------------------------------------------------------------------
//  CPU reference: SpMV from ELL arrays (CPU-side — simple iteration)
// ---------------------------------------------------------------------------
void host_spmv_ell(const float *value, const int *colIdx,
                   int numRows, int numColsPerRow,
                   const float *x, float *y) {
    for (int row = 0; row < numRows; row++) {
        float sum = 0.0f;
        for (int t = 0; t < numColsPerRow; t++) {
            int idx = t * numRows + row;
            int col = colIdx[idx];
            float val = value[idx];
            sum += val * x[col];
        }
        y[row] = sum;
    }
}

// ---------------------------------------------------------------------------
//  Helper: build ELL arrays from a dense matrix
// ---------------------------------------------------------------------------
void dense_to_ell(const float *dense, int rows, int cols,
                  float *&value, int *&colIdx, int *&nnzPerRow,
                  int &numColsPerRow) {
    // Count nonzeros per row
    nnzPerRow = new int[rows]();
    int max_nnz = 0;
    for (int r = 0; r < rows; r++) {
        int cnt = 0;
        for (int c = 0; c < cols; c++) {
            if (std::fabs(dense[r * cols + c]) > 1e-6f)
                cnt++;
        }
        nnzPerRow[r] = cnt;
        max_nnz = std::max(max_nnz, cnt);
    }
    numColsPerRow = max_nnz;

    // Allocate column-major storage: numColsPerRow * rows
    value  = new float[numColsPerRow * rows]();   // zero-initialized (padding)
    colIdx = new int[numColsPerRow * rows]();

    // Fill in column-major order
    for (int t = 0; t < max_nnz; t++) {
        for (int r = 0; r < rows; r++) {
            // Find the t-th nonzero in row r
            int nnz_found = 0;
            bool found = false;
            for (int c = 0; c < cols; c++) {
                if (std::fabs(dense[r * cols + c]) > 1e-6f) {
                    if (nnz_found == t) {
                        int idx = t * rows + r;
                        value[idx]  = dense[r * cols + c];
                        colIdx[idx] = c;
                        found = true;
                        break;
                    }
                    nnz_found++;
                }
            }
            if (!found) {
                // Padding element — value is 0 (already zero-initialized)
                int idx = t * rows + r;
                colIdx[idx] = 0;  // colIdx doesn't matter when value=0
            }
        }
    }
}

// ---------------------------------------------------------------------------
//  Main
// ---------------------------------------------------------------------------
int main() {
    CHECK_CUDA(cudaSetDevice(1));
    print_device_info(1);

    printf("╔══════════════════════════════════════════════════════════╗\n");
    printf("║  SpMV/ELL — Sparse Matrix-Vector Multiplication          ║\n");
    printf("║  ELL (ELLPACK) format — column-major, coalesced          ║\n");
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
    float h_y_gpu1[4] = {0};

    float *h_value1 = nullptr;
    int   *h_colIdx1 = nullptr, *h_nnzPerRow1 = nullptr;
    int    numColsPerRow1 = 0;
    dense_to_ell(dense1, rows1, cols1, h_value1, h_colIdx1, h_nnzPerRow1,
                 numColsPerRow1);

    printf("  Rows: %d, Max nonzeros/row: %d\n", rows1, numColsPerRow1);
    printf("  nnzPerRow: [%d, %d, %d, %d]\n",
           h_nnzPerRow1[0], h_nnzPerRow1[1],
           h_nnzPerRow1[2], h_nnzPerRow1[3]);

    // CPU reference (dense)
    for (int r = 0; r < rows1; r++)
        for (int c = 0; c < cols1; c++)
            h_y_ref1[r] += dense1[r * cols1 + c] * h_x1[c];

    // Device allocations
    int ell_size1 = numColsPerRow1 * rows1;
    float *d_value1, *d_x1, *d_y1;
    int   *d_colIdx1;
    CHECK_CUDA(cudaMalloc(&d_value1,   ell_size1 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_colIdx1,  ell_size1 * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_x1,       cols1 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_y1,       rows1 * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_value1,  h_value1,  ell_size1 * sizeof(float),
                           cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_colIdx1, h_colIdx1, ell_size1 * sizeof(int),
                           cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_x1,      h_x1,      cols1 * sizeof(float),
                           cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_y1, 0, rows1 * sizeof(float)));

    int blockSize = 256;
    int gridSize  = (rows1 + blockSize - 1) / blockSize;

    spmv_ell_kernel<<<gridSize, blockSize>>>(d_value1, d_colIdx1,
                                             rows1, numColsPerRow1,
                                             d_x1, d_y1);
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
    delete[] h_nnzPerRow1;
    CHECK_CUDA(cudaFree(d_value1));
    CHECK_CUDA(cudaFree(d_colIdx1));
    CHECK_CUDA(cudaFree(d_x1));
    CHECK_CUDA(cudaFree(d_y1));

    // -----------------------------------------------------------------------
    //  Test 2: Random 4096×4096 sparse matrix (~0.5% density)
    // -----------------------------------------------------------------------
    printf("--- Test 2: Random 4096×4096 sparse matrix (~0.5%% density) ---\n");

    const int rows2 = 4096, cols2 = 4096;
    const float density = 0.005f;

    // Generate a dense matrix (very sparse — store sparsely was the whole point,
    // but for reference we need it; let's generate small dense and convert)
    float *h_dense2 = new float[rows2 * cols2]();
    int *h_nnzPerRow2 = new int[rows2]();

    srand(234);
    int total_nnz2 = 0;
    for (int r = 0; r < rows2; r++) {
        int row_nnz = 0;
        for (int c = 0; c < cols2; c++) {
            if ((rand() % 1000) < (int)(density * 1000)) {
                h_dense2[r * cols2 + c] = (float)(rand() % 20) / 5.0f + 0.1f;
                row_nnz++;
                total_nnz2++;
            }
        }
        if (row_nnz == 0) {
            int c = rand() % cols2;
            h_dense2[r * cols2 + c] = 1.0f;
            row_nnz = 1;
            total_nnz2++;
        }
        h_nnzPerRow2[r] = row_nnz;
    }

    // Build ELL arrays
    float *h_value2 = nullptr;
    int   *h_colIdx2 = nullptr, *h_nnzPerRow2_dummy = nullptr;
    int    numColsPerRow2 = 0;
    dense_to_ell(h_dense2, rows2, cols2, h_value2, h_colIdx2,
                 h_nnzPerRow2_dummy, numColsPerRow2);

    printf("  Total nonzeros: %d, Max nonzeros/row: %d, ELL size: %d\n",
           total_nnz2, numColsPerRow2, numColsPerRow2 * rows2);
    printf("  ELL padding overhead: %.1f%%\n",
           (float)(numColsPerRow2 * rows2 - total_nnz2) / total_nnz2 * 100.0f);

    float *h_x2 = new float[cols2];
    float *h_y_ref2 = new float[rows2]();
    float *h_y_gpu2 = new float[rows2]();

    for (int i = 0; i < cols2; i++)
        h_x2[i] = (float)(rand() % 100) / 25.0f;

    // CPU reference (dense)
    for (int r = 0; r < rows2; r++)
        for (int c = 0; c < cols2; c++)
            h_y_ref2[r] += h_dense2[r * cols2 + c] * h_x2[c];

    // Device
    int ell_size2 = numColsPerRow2 * rows2;
    float *d_value2, *d_x2, *d_y2;
    int   *d_colIdx2;
    CHECK_CUDA(cudaMalloc(&d_value2,   ell_size2 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_colIdx2,  ell_size2 * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_x2,       cols2 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_y2,       rows2 * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_value2,  h_value2,  ell_size2 * sizeof(float),
                           cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_colIdx2, h_colIdx2, ell_size2 * sizeof(int),
                           cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_x2,      h_x2,      cols2 * sizeof(float),
                           cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_y2, 0, rows2 * sizeof(float)));

    gridSize = (rows2 + blockSize - 1) / blockSize;

    // Warm-up
    spmv_ell_kernel<<<gridSize, blockSize>>>(d_value2, d_colIdx2,
                                             rows2, numColsPerRow2,
                                             d_x2, d_y2);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemset(d_y2, 0, rows2 * sizeof(float)));

    // Timed run
    gpu_timer timer;
    timer.start();
    spmv_ell_kernel<<<gridSize, blockSize>>>(d_value2, d_colIdx2,
                                             rows2, numColsPerRow2,
                                             d_x2, d_y2);
    timer.stop();
    float ms2 = timer.elapsed_ms();

    CHECK_CUDA(cudaMemcpy(h_y_gpu2, d_y2, rows2 * sizeof(float),
                           cudaMemcpyDeviceToHost));

    bool pass2 = cpu_allclose(h_y_ref2, h_y_gpu2, rows2);
    printf("  Kernel time: %.3f ms\n", ms2);
    printf("  Throughput:  %.2f M nonzeros/sec\n",
           total_nnz2 / ms2 / 1000.0f);
    printf("  %s\n\n", pass2 ? "VALIDATION: PASS" : "VALIDATION: FAIL");

    // Also test the nnzPerRow variant
    printf("--- Test 3: ELL kernel with nnzPerRow (same data) ---\n");
    int *d_nnzPerRow2;
    CHECK_CUDA(cudaMalloc(&d_nnzPerRow2, rows2 * sizeof(int)));
    CHECK_CUDA(cudaMemcpy(d_nnzPerRow2, h_nnzPerRow2, rows2 * sizeof(int),
                           cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_y2, 0, rows2 * sizeof(float)));

    spmv_ell_nnz_kernel<<<gridSize, blockSize>>>(d_value2, d_colIdx2,
                                                  d_nnzPerRow2,
                                                  rows2, numColsPerRow2,
                                                  d_x2, d_y2);
    CHECK_CUDA(cudaDeviceSynchronize());

    float *h_y_nnz = new float[rows2];
    CHECK_CUDA(cudaMemcpy(h_y_nnz, d_y2, rows2 * sizeof(float),
                           cudaMemcpyDeviceToHost));

    bool pass3 = cpu_allclose(h_y_ref2, h_y_nnz, rows2);
    printf("  %s\n", pass3 ? "VALIDATION: PASS" : "VALIDATION: FAIL");

    delete[] h_dense2;
    delete[] h_nnzPerRow2;
    delete[] h_value2;
    delete[] h_colIdx2;
    delete[] h_nnzPerRow2_dummy;
    delete[] h_x2;
    delete[] h_y_ref2;
    delete[] h_y_gpu2;
    delete[] h_y_nnz;
    CHECK_CUDA(cudaFree(d_value2));
    CHECK_CUDA(cudaFree(d_colIdx2));
    CHECK_CUDA(cudaFree(d_nnzPerRow2));
    CHECK_CUDA(cudaFree(d_x2));
    CHECK_CUDA(cudaFree(d_y2));

    printf("\nDone.\n");
    return (pass1 && pass2 && pass3) ? 0 : 1;
}
