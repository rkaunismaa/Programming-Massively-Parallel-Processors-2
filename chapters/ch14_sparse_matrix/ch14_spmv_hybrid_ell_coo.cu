/*
 * =============================================================================
 *  ch14_spmv_hybrid_ell_coo.cu — Hybrid ELL-COO SpMV
 *
 *  Book:      Programming Massively Parallel Processors (4th Ed.)
 *  Section:   14.5 — Hybrid ELL-COO format (Fig 14.13)
 *  Exercise:  4 — Host code for producing hybrid ELL-COO and using it
 *
 *  Summary:
 *    When some rows have far more nonzeros than others, the ELL format
 *    wastes space on padding. Hybrid ELL-COO caps ELL at K nonzeros per
 *    row and stores overflow elements in a COO format.
 *
 *    This implementation:
 *      1. Converts a dense matrix → hybrid ELL-COO with threshold K
 *      2. Runs SpMV/ELL kernel for the ELL part
 *      3. Runs SpMV/COO kernel for the COO overflow part
 *      4. Combines results
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
#include <numeric>
#include <vector>

// ---------------------------------------------------------------------------
//  SpMV/ELL kernel (same as Fig 14.12)
// ---------------------------------------------------------------------------
__global__ void spmv_ell_kernel(const float *value, const int *colIdx,
                                int numRows, int numColsPerRow,
                                const float *x, float *y) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < numRows) {
        float sum = 0.0f;
        for (int t = 0; t < numColsPerRow; t++) {
            int idx = t * numRows + row;
            sum += value[idx] * x[colIdx[idx]];
        }
        y[row] = sum;
    }
}

// ---------------------------------------------------------------------------
//  SpMV/COO kernel (same as Fig 14.5)
// ---------------------------------------------------------------------------
__global__ void spmv_coo_kernel(const float *value, const int *colIdx,
                                const int *rowIdx, const float *x,
                                float *y, int numNonzeros) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numNonzeros) {
        atomicAdd(&y[rowIdx[i]], value[i] * x[colIdx[i]]);
    }
}

// ---------------------------------------------------------------------------
//  CPU reference: dense SpMV
// ---------------------------------------------------------------------------
void host_spmv_dense(const float *A, const float *x, float *y,
                     int rows, int cols) {
    for (int r = 0; r < rows; r++) {
        y[r] = 0.0f;
        for (int c = 0; c < cols; c++)
            y[r] += A[r * cols + c] * x[c];
    }
}

// ---------------------------------------------------------------------------
//  Main
// ---------------------------------------------------------------------------
int main() {
    CHECK_CUDA(cudaSetDevice(1));
    print_device_info(1);

    printf("╔══════════════════════════════════════════════════════════╗\n");
    printf("║  Hybrid ELL-COO SpMV                                     ║\n");
    printf("║  ELL for regular rows, COO for overflow elements          ║\n");
    printf("╚══════════════════════════════════════════════════════════╝\n\n");

    // -----------------------------------------------------------------------
    //  Test: Build a matrix with skewed row distribution
    //  Row 0: 20 nonzeros, Rows 1-127: ~3 nonzeros each
    //  Threshold K = 4 → ELL stores 4 per row, COO gets Row 0's overflow
    // -----------------------------------------------------------------------
    printf("--- Test: Skewed matrix (1 long row + regular rows) ---\n");

    const int rows = 512;
    const int cols = 512;

    // Generate a dense matrix with one long row
    float *h_dense = new float[rows * cols]();
    srand(345);

    // Row 0: 20 nonzeros to demonstrate skew
    for (int c = 0; c < 20; c++)
        h_dense[0 * cols + (c * 25) % cols] = (float)(rand() % 10) / 3.0f + 0.5f;

    // Rows 1..rows-1: ~3 nonzeros each
    for (int r = 1; r < rows; r++) {
        int nnz = 0;
        for (int c = 0; c < cols; c++) {
            if ((rand() % 200) < 3) {
                h_dense[r * cols + c] = (float)(rand() % 10) / 3.0f + 0.5f;
                nnz++;
            }
        }
        if (nnz == 0) {
            h_dense[r * cols + rand() % cols] = 1.0f;
        }
    }

    float *h_x = new float[cols];
    for (int i = 0; i < cols; i++)
        h_x[i] = (float)(rand() % 100) / 25.0f;

    float *h_y_ref = new float[rows]();
    host_spmv_dense(h_dense, h_x, h_y_ref, rows, cols);

    // -----------------------------------------------------------------------
    //  Build hybrid ELL-COO format
    //  Threshold K = 4 — ELL stores at most 4 nonzeros per row
    // -----------------------------------------------------------------------
    const int K = 4;
    int numColsPerRow = K;

    // Count nonzeros per row and total for COO overflow
    std::vector<int> row_nnz(rows, 0);
    int total_coo = 0;
    int max_longest = 0;

    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            if (std::fabs(h_dense[r * cols + c]) > 1e-6f)
                row_nnz[r]++;
        }
        if (row_nnz[r] > K)
            total_coo += row_nnz[r] - K;
        max_longest = std::max(max_longest, row_nnz[r]);
    }

    printf("  Matrix: %d x %d\n", rows, cols);
    printf("  Row 0 nonzeros: %d\n", row_nnz[0]);
    printf("  Average row nonzeros: %.1f\n",
           (float)std::accumulate(row_nnz.begin(), row_nnz.end(), 0) / rows);
    printf("  ELL threshold K: %d\n", K);
    printf("  ELL max/row: %d\n", numColsPerRow);
    printf("  COO overflow elements: %d\n", total_coo);

    // Build ELL part (column-major, K entries per row)
    float *h_ell_value  = new float[numColsPerRow * rows]();
    int   *h_ell_colIdx = new int[numColsPerRow * rows]();
    std::vector<int> ell_count(rows, 0);  // how many placed in ELL per row

    for (int r = 0; r < rows; r++) {
        int placed = 0;
        for (int c = 0; c < cols && placed < K; c++) {
            if (std::fabs(h_dense[r * cols + c]) > 1e-6f) {
                int idx = placed * rows + r;  // column-major
                h_ell_value[idx]  = h_dense[r * cols + c];
                h_ell_colIdx[idx] = c;
                placed++;
            }
        }
        ell_count[r] = placed;
    }

    // Build COO part (overflow)
    float *h_coo_value  = new float[total_coo]();
    int   *h_coo_colIdx = new int[total_coo]();
    int   *h_coo_rowIdx = new int[total_coo]();
    int coo_idx = 0;

    for (int r = 0; r < rows; r++) {
        int taken = 0;
        for (int c = 0; c < cols; c++) {
            if (std::fabs(h_dense[r * cols + c]) > 1e-6f) {
                if (taken >= K) {
                    h_coo_value[coo_idx]  = h_dense[r * cols + c];
                    h_coo_colIdx[coo_idx] = c;
                    h_coo_rowIdx[coo_idx] = r;
                    coo_idx++;
                }
                taken++;
            }
        }
    }

    printf("  Actual COO elements stored: %d\n", coo_idx);
    printf("  ELL storage: %d floats + %d ints\n",
           numColsPerRow * rows, numColsPerRow * rows);
    printf("  Versus full CSR storage: ~%d floats + %d ints\n",
           std::accumulate(row_nnz.begin(), row_nnz.end(), 0),
           std::accumulate(row_nnz.begin(), row_nnz.end(), 0));

    // -----------------------------------------------------------------------
    //  Device allocations
    // -----------------------------------------------------------------------
    int blockSize = 256;
    float *d_x, *d_y;
    CHECK_CUDA(cudaMalloc(&d_x, cols * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_y, rows * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_x, h_x, cols * sizeof(float),
                           cudaMemcpyHostToDevice));

    // ELL part on device
    int ell_slots = numColsPerRow * rows;
    float *d_ell_value;
    int   *d_ell_colIdx;
    CHECK_CUDA(cudaMalloc(&d_ell_value, ell_slots * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_ell_colIdx, ell_slots * sizeof(int)));
    CHECK_CUDA(cudaMemcpy(d_ell_value, h_ell_value, ell_slots * sizeof(float),
                           cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_ell_colIdx, h_ell_colIdx, ell_slots * sizeof(int),
                           cudaMemcpyHostToDevice));

    // COO part on device
    float *d_coo_value;
    int   *d_coo_colIdx, *d_coo_rowIdx;
    CHECK_CUDA(cudaMalloc(&d_coo_value,  total_coo * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_coo_colIdx, total_coo * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_coo_rowIdx, total_coo * sizeof(int)));
    CHECK_CUDA(cudaMemcpy(d_coo_value, h_coo_value, total_coo * sizeof(float),
                           cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_coo_colIdx, h_coo_colIdx, total_coo * sizeof(int),
                           cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_coo_rowIdx, h_coo_rowIdx, total_coo * sizeof(int),
                           cudaMemcpyHostToDevice));

    // -----------------------------------------------------------------------
    //  Run ELL kernel
    // -----------------------------------------------------------------------
    int gridSize_ell = (rows + blockSize - 1) / blockSize;

    CHECK_CUDA(cudaMemset(d_y, 0, rows * sizeof(float)));
    spmv_ell_kernel<<<gridSize_ell, blockSize>>>(d_ell_value, d_ell_colIdx,
                                                 rows, numColsPerRow,
                                                 d_x, d_y);
    CHECK_CUDA(cudaDeviceSynchronize());

    // -----------------------------------------------------------------------
    //  Run COO kernel (accumulates into same y)
    // -----------------------------------------------------------------------
    if (total_coo > 0) {
        int gridSize_coo = (total_coo + blockSize - 1) / blockSize;
        spmv_coo_kernel<<<gridSize_coo, blockSize>>>(d_coo_value, d_coo_colIdx,
                                                     d_coo_rowIdx, d_x, d_y,
                                                     total_coo);
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    // -----------------------------------------------------------------------
    //  Validate
    // -----------------------------------------------------------------------
    float *h_y_gpu = new float[rows]();
    CHECK_CUDA(cudaMemcpy(h_y_gpu, d_y, rows * sizeof(float),
                           cudaMemcpyDeviceToHost));

    printf("\n  Validation against dense CPU reference:\n");
    bool pass = cpu_allclose(h_y_ref, h_y_gpu, rows);
    printf("  %s\n\n", pass ? "VALIDATION: PASS" : "VALIDATION: FAIL");

    // -----------------------------------------------------------------------
    //  Compare with full ELL (no hybrid) to show overhead
    // -----------------------------------------------------------------------
    printf("--- Comparison: Full ELL (K = max nnz = %d) ---\n", max_longest);

    float *h_full_ell_value  = new float[max_longest * rows]();
    int   *h_full_ell_colIdx = new int[max_longest * rows]();

    // Build full ELL
    for (int r = 0; r < rows; r++) {
        int placed = 0;
        for (int c = 0; c < cols && placed < max_longest; c++) {
            if (std::fabs(h_dense[r * cols + c]) > 1e-6f) {
                int idx = placed * rows + r;
                h_full_ell_value[idx]  = h_dense[r * cols + c];
                h_full_ell_colIdx[idx] = c;
                placed++;
            }
        }
    }

    printf("  Full ELL storage: %d slots (vs hybrid %d + %d COO)\n",
           max_longest * rows, ell_slots, total_coo);
    printf("  Full-ELL padding overhead: %d elements (%.0f%%)\n",
           max_longest * rows - std::accumulate(row_nnz.begin(), row_nnz.end(), 0),
           (float)(max_longest * rows) /
               std::accumulate(row_nnz.begin(), row_nnz.end(), 0) * 100.0f - 100.0f);

    // Cleanup
    delete[] h_dense;
    delete[] h_x;
    delete[] h_y_ref;
    delete[] h_y_gpu;
    delete[] h_ell_value;
    delete[] h_ell_colIdx;
    delete[] h_coo_value;
    delete[] h_coo_colIdx;
    delete[] h_coo_rowIdx;
    delete[] h_full_ell_value;
    delete[] h_full_ell_colIdx;

    CHECK_CUDA(cudaFree(d_x));
    CHECK_CUDA(cudaFree(d_y));
    CHECK_CUDA(cudaFree(d_ell_value));
    CHECK_CUDA(cudaFree(d_ell_colIdx));
    CHECK_CUDA(cudaFree(d_coo_value));
    CHECK_CUDA(cudaFree(d_coo_colIdx));
    CHECK_CUDA(cudaFree(d_coo_rowIdx));

    printf("\nDone.\n");
    return pass ? 0 : 1;
}
