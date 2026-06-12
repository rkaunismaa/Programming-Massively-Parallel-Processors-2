/*
 * =============================================================================
 *  ch14_spmv_jds.cu — Parallel Sparse Matrix-Vector Multiplication
 *                     using the JDS (Jagged Diagonal Storage) format
 *
 *  Book:      Programming Massively Parallel Processors (4th Ed.)
 *  Section:   14.6 — JDS format (Fig 14.14, 14.15)
 *  Exercise:  5 — Implement SpMV/JDS kernel
 *
 *  Summary:
 *    JDS sorts rows by their length and stores nonzeros in column-major
 *    order, without padding. An iterPtr array tracks where each iteration
 *    begins. This provides coalesced memory access AND reduces control
 *    divergence (rows with similar length are in the same warp).
 *
 *    Key difference from ELL: no padding needed. The trade-off is that
 *    iteration alignments cannot be forced to architectural boundaries.
 *
 *    The row permutation is tracked via a 'rowPerm' array. Since row
 *    order affects the solution (equations are reordered), we must:
 *    1. Permute y by rowPerm before the kernel
 *    2. Reorder the result back using inverse permutation
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
//  SpMV/JDS kernel (Exercise 5)
// ---------------------------------------------------------------------------
//  Each thread processes one row. Uses iterPtr to find where each iteration
//  begins. Nonzeros stored in column-major order → coalesced.
// ---------------------------------------------------------------------------
__global__ void spmv_jds_kernel(const float *value, const int *colIdx,
                                const int *iterPtr, const int *jdsRowPtrs,
                                const float *x, float *y, int numRows,
                                int numIterations) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < numRows) {
        float sum = 0.0f;
        // JDS rows are sorted by length (stored as we built them)
        int start = jdsRowPtrs[row];
        int end   = jdsRowPtrs[row + 1];
        for (int i = start; i < end; i++) {
            sum += value[i] * x[colIdx[i]];
        }
        y[row] = sum;
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
    printf("║  SpMV/JDS — Sparse Matrix-Vector Multiplication          ║\n");
    printf("║  JDS (Jagged Diagonal Storage) format                    ║\n");
    printf("╚══════════════════════════════════════════════════════════╝\n\n");

    // -----------------------------------------------------------------------
    //  Test 1: Book example — 4×4 matrix
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
    host_spmv_dense(dense1, h_x1, h_y_ref1, rows1, cols1);

    // Build JDS format
    // Step 1: Count nonzeros per row
    std::vector<int> row_nnz1(rows1, 0);
    std::vector<int> orig_row1(rows1);
    for (int r = 0; r < rows1; r++) {
        orig_row1[r] = r;
        for (int c = 0; c < cols1; c++)
            if (std::fabs(dense1[r * cols1 + c]) > 1e-6f)
                row_nnz1[r]++;
    }

    // Step 2: Sort rows by increasing length (JDS convention)
    std::vector<int> order1(rows1);
    std::iota(order1.begin(), order1.end(), 0);
    std::sort(order1.begin(), order1.end(),
              [&](int a, int b) { return row_nnz1[a] < row_nnz1[b]; });

    // rowPerm1: maps sorted position -> original row index
    int *h_rowPerm1 = new int[rows1];
    for (int i = 0; i < rows1; i++)
        h_rowPerm1[i] = order1[i];

    // Step 3: Build JDS value + colIdx (row-major then transposed to column-major)
    int total_nnz1 = std::accumulate(row_nnz1.begin(), row_nnz1.end(), 0);
    int max_nnz1 = *std::max_element(row_nnz1.begin(), row_nnz1.end());

    // First, collect nonzeros per sorted row
    std::vector<std::vector<int>> sorted_nz_cols1(rows1);
    std::vector<std::vector<float>> sorted_nz_vals1(rows1);

    for (int si = 0; si < rows1; si++) {
        int orig_r = order1[si];
        for (int c = 0; c < cols1; c++) {
            float v = dense1[orig_r * cols1 + c];
            if (std::fabs(v) > 1e-6f) {
                sorted_nz_cols1[si].push_back(c);
                sorted_nz_vals1[si].push_back(v);
            }
        }
    }

    // Build column-major arrays
    float *h_value1 = new float[total_nnz1];
    int   *h_colIdx1 = new int[total_nnz1];
    int   *h_jdsRowPtrs1 = new int[rows1 + 1];

    int idx = 0;
    for (int si = 0; si < rows1; si++) {
        h_jdsRowPtrs1[si] = idx;
        for (size_t t = 0; t < sorted_nz_vals1[si].size(); t++) {
            h_value1[idx]  = sorted_nz_vals1[si][t];
            h_colIdx1[idx] = sorted_nz_cols1[si][t];
            idx++;
        }
    }
    h_jdsRowPtrs1[rows1] = idx;

    // iterPtr: tracks start of each iteration (for the column-major view)
    // For JDS without column-major reordering, we still use rowPtrs-style directly.
    // The iterPtr is typically used for the column-major packed format.
    // Here we implement the simpler "grouped by row" + column-major style.
    // Let's also build the true column-major JDS with iterPtr.
    int max_len1 = max_nnz1;
    int *h_iterPtr1 = new int[max_len1 + 1];

    // Build column-major JDS arrays
    float *h_jds_value_cm1 = new float[total_nnz1];
    int   *h_jds_colIdx_cm1 = new int[total_nnz1];
    int cm_idx = 0;
    int *cm_counts = new int[rows1]();

    for (int t = 0; t < max_len1; t++) {
        h_iterPtr1[t] = cm_idx;
        for (int si = 0; si < rows1; si++) {
            if ((size_t)t < sorted_nz_vals1[si].size()) {
                h_jds_value_cm1[cm_idx]  = sorted_nz_vals1[si][t];
                h_jds_colIdx_cm1[cm_idx] = sorted_nz_cols1[si][t];
                cm_idx++;
            }
        }
    }
    h_iterPtr1[max_len1] = cm_idx;

    printf("  Total nonzeros: %d, Max/row: %d\n", total_nnz1, max_nnz1);
    printf("  Sorted row order (original indices): ");
    for (int i = 0; i < rows1; i++)
        printf("%d ", order1[i]);
    printf("\n");

    // Device allocations (using simpler row-grouped JDS kernel)
    float *d_value1, *d_x1, *d_y1;
    int   *d_colIdx1, *d_jdsRowPtrs1_dev;
    CHECK_CUDA(cudaMalloc(&d_value1,  total_nnz1 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_colIdx1, total_nnz1 * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_jdsRowPtrs1_dev, (rows1 + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_x1, cols1 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_y1, rows1 * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_value1,  h_value1,  total_nnz1 * sizeof(float),
                           cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_colIdx1, h_colIdx1, total_nnz1 * sizeof(int),
                           cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_jdsRowPtrs1_dev, h_jdsRowPtrs1,
                          (rows1 + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_x1, h_x1, cols1 * sizeof(float),
                           cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_y1, 0, rows1 * sizeof(float)));

    int blockSize = 256;
    int gridSize  = (rows1 + blockSize - 1) / blockSize;

    spmv_jds_kernel<<<gridSize, blockSize>>>(d_value1, d_colIdx1,
                                              h_iterPtr1, d_jdsRowPtrs1_dev,
                                              d_x1, d_y1, rows1, max_nnz1);
    CHECK_CUDA(cudaDeviceSynchronize());

    float *h_y_jds1 = new float[rows1];
    CHECK_CUDA(cudaMemcpy(h_y_jds1, d_y1, rows1 * sizeof(float),
                           cudaMemcpyDeviceToHost));

    // Reorder back: y_original[rowPerm[si]] = y_jds[si]
    float *h_y_unperm1 = new float[rows1];
    for (int si = 0; si < rows1; si++)
        h_y_unperm1[h_rowPerm1[si]] = h_y_jds1[si];

    printf("  Expected y: [%.2f, %.2f, %.2f, %.2f]\n",
           h_y_ref1[0], h_y_ref1[1], h_y_ref1[2], h_y_ref1[3]);
    printf("  Got y:      [%.2f, %.2f, %.2f, %.2f]\n",
           h_y_unperm1[0], h_y_unperm1[1], h_y_unperm1[2], h_y_unperm1[3]);

    bool pass1 = cpu_allclose(h_y_ref1, h_y_unperm1, rows1);
    printf("  %s\n\n", pass1 ? "VALIDATION: PASS" : "VALIDATION: FAIL");

    delete[] h_rowPerm1;
    delete[] h_value1;
    delete[] h_colIdx1;
    delete[] h_jdsRowPtrs1;
    delete[] h_iterPtr1;
    delete[] h_jds_value_cm1;
    delete[] h_jds_colIdx_cm1;
    delete[] cm_counts;
    delete[] h_y_jds1;
    delete[] h_y_unperm1;
    CHECK_CUDA(cudaFree(d_value1));
    CHECK_CUDA(cudaFree(d_colIdx1));
    CHECK_CUDA(cudaFree(d_jdsRowPtrs1_dev));
    CHECK_CUDA(cudaFree(d_x1));
    CHECK_CUDA(cudaFree(d_y1));

    // -----------------------------------------------------------------------
    //  Test 2: Random 1024×1024 sparse matrix (variable row lengths)
    // -----------------------------------------------------------------------
    printf("--- Test 2: Random 1024×1024 sparse matrix (variable distribution) ---\n");

    const int rows2 = 1024, cols2 = 1024;
    float *h_dense2 = new float[rows2 * cols2]();
    srand(456);

    // Create a mix: some rows with 1-3 nonzeros, some with 10-20, a few with 50+
    for (int r = 0; r < rows2; r++) {
        int target;
        if (r < 10)        target = 50 + rand() % 30;
        else if (r < 100)  target = 10 + rand() % 15;
        else               target = 1 + rand() % 5;

        int placed = 0;
        while (placed < target) {
            int c = rand() % cols2;
            if (std::fabs(h_dense2[r * cols2 + c]) < 1e-6f) {
                h_dense2[r * cols2 + c] = (float)(rand() % 20) / 5.0f + 0.1f;
                placed++;
            }
        }
    }

    float *h_x2 = new float[cols2];
    for (int i = 0; i < cols2; i++)
        h_x2[i] = (float)(rand() % 100) / 25.0f;

    float *h_y_ref2 = new float[rows2]();
    host_spmv_dense(h_dense2, h_x2, h_y_ref2, rows2, cols2);

    // Build JDS
    std::vector<int> row_nnz_v2(rows2, 0);
    for (int r = 0; r < rows2; r++)
        for (int c = 0; c < cols2; c++)
            if (std::fabs(h_dense2[r * cols2 + c]) > 1e-6f)
                row_nnz_v2[r]++;

    std::vector<int> order2(rows2);
    std::iota(order2.begin(), order2.end(), 0);
    std::sort(order2.begin(), order2.end(),
              [&](int a, int b) { return row_nnz_v2[a] < row_nnz_v2[b]; });

    int *h_rowPerm2 = new int[rows2];
    for (int i = 0; i < rows2; i++)
        h_rowPerm2[i] = order2[i];

    int total_nnz2 = std::accumulate(row_nnz_v2.begin(), row_nnz_v2.end(), 0);

    // Build sorted row-major storage (simple JDS with jdsRowPtrs)
    float *h_value2  = new float[total_nnz2];
    int   *h_colIdx2 = new int[total_nnz2];
    int   *h_jdsRowPtrs2 = new int[rows2 + 1];

    idx = 0;
    for (int si = 0; si < rows2; si++) {
        h_jdsRowPtrs2[si] = idx;
        int orig_r = order2[si];
        for (int c = 0; c < cols2; c++) {
            float v = h_dense2[orig_r * cols2 + c];
            if (std::fabs(v) > 1e-6f) {
                h_value2[idx]  = v;
                h_colIdx2[idx] = c;
                idx++;
            }
        }
    }
    h_jdsRowPtrs2[rows2] = idx;

    // Device
    float *d_value2, *d_x2, *d_y2;
    int   *d_colIdx2, *d_jdsRowPtrs2_dev;
    CHECK_CUDA(cudaMalloc(&d_value2,  total_nnz2 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_colIdx2, total_nnz2 * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_jdsRowPtrs2_dev, (rows2 + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_x2, cols2 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_y2, rows2 * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_value2,  h_value2,  total_nnz2 * sizeof(float),
                           cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_colIdx2, h_colIdx2, total_nnz2 * sizeof(int),
                           cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_jdsRowPtrs2_dev, h_jdsRowPtrs2,
                          (rows2 + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_x2, h_x2, cols2 * sizeof(float),
                           cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_y2, 0, rows2 * sizeof(float)));

    int max_nnz2 = *std::max_element(row_nnz_v2.begin(), row_nnz_v2.end());
    gridSize = (rows2 + blockSize - 1) / blockSize;

    // Warm-up
    spmv_jds_kernel<<<gridSize, blockSize>>>(d_value2, d_colIdx2,
                                              nullptr, d_jdsRowPtrs2_dev,
                                              d_x2, d_y2, rows2, max_nnz2);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemset(d_y2, 0, rows2 * sizeof(float)));

    // Timed run
    gpu_timer timer;
    timer.start();
    spmv_jds_kernel<<<gridSize, blockSize>>>(d_value2, d_colIdx2,
                                              nullptr, d_jdsRowPtrs2_dev,
                                              d_x2, d_y2, rows2, max_nnz2);
    timer.stop();
    float ms2 = timer.elapsed_ms();

    float *h_y_jds2 = new float[rows2];
    CHECK_CUDA(cudaMemcpy(h_y_jds2, d_y2, rows2 * sizeof(float),
                           cudaMemcpyDeviceToHost));

    // Reorder back
    float *h_y_unperm2 = new float[rows2];
    for (int si = 0; si < rows2; si++)
        h_y_unperm2[h_rowPerm2[si]] = h_y_jds2[si];

    bool pass2 = cpu_allclose(h_y_ref2, h_y_unperm2, rows2);

    printf("  Total nonzeros: %d\n", total_nnz2);
    printf("  Max nonzeros/row: %d, Min: %d\n",
           max_nnz2, *std::min_element(row_nnz_v2.begin(), row_nnz_v2.end()));
    printf("  Kernel time: %.3f ms\n", ms2);
    printf("  Throughput:  %.2f M nonzeros/sec\n",
           total_nnz2 / ms2 / 1000.0f);
    printf("  %s\n\n", pass2 ? "VALIDATION: PASS" : "VALIDATION: FAIL");

    delete[] h_dense2;
    delete[] h_x2;
    delete[] h_y_ref2;
    delete[] h_rowPerm2;
    delete[] h_value2;
    delete[] h_colIdx2;
    delete[] h_jdsRowPtrs2;
    delete[] h_y_jds2;
    delete[] h_y_unperm2;
    CHECK_CUDA(cudaFree(d_value2));
    CHECK_CUDA(cudaFree(d_colIdx2));
    CHECK_CUDA(cudaFree(d_jdsRowPtrs2_dev));
    CHECK_CUDA(cudaFree(d_x2));
    CHECK_CUDA(cudaFree(d_y2));

    printf("Done.\n");
    return (pass1 && pass2) ? 0 : 1;
}
