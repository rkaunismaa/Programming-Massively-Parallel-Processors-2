/*
 * =============================================================================
 *  ch14_coo_to_csr.cu — Exercise 3: COO → CSR Conversion
 *
 *  Book:      Programming Massively Parallel Processors (4th Ed.)
 *  Exercise:  3 — Convert from COO to CSR using histogram and prefix sum
 *
 *  Summary:
 *    Converting COO to CSR requires:
 *      1. Histogram: count nonzeros per row from rowIdx[]
 *      2. Exclusive prefix sum: compute rowPtrs from histogram
 *      3. Scatter: place each nonzero into the correct CSR position
 *         using rowPtrs and per-row position counters
 *
 *    The textbook notes: "Converting from COO to CSR on the GPU is an
 *    excellent exercise for the reader, using multiple fundamental
 *    parallel computing primitives, including histogram and prefix sum."
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
//  Step 1: Parallel histogram (count nonzeros per row)
//  We use a privatized-shared-memory approach (Ch9 pattern).
// ---------------------------------------------------------------------------
__global__ void histogram_row_kernel(const int *rowIdx, int *hist,
                                     int numNonzeros, int numRows) {
    // Privatized shared-memory histogram
    extern __shared__ int s_hist[];
    // All threads participate in zeroing ALL bins (stride-based)
    for (int i = threadIdx.x; i < numRows; i += blockDim.x)
        s_hist[i] = 0;
    __syncthreads();

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (; i < numNonzeros; i += stride) {
        int row = rowIdx[i];
        if (row < numRows)
            atomicAdd(&s_hist[row], 1);
    }
    __syncthreads();

    // All threads merge their shared histogram to global (stride-based)
    for (int i = threadIdx.x; i < numRows; i += blockDim.x) {
        if (s_hist[i] > 0)
            atomicAdd(&hist[i], s_hist[i]);
    }
}

// ---------------------------------------------------------------------------
//  Step 2: Exclusive prefix sum on histogram → rowPtrs
//  We use a simple Kogge-Stone scan (Ch11 pattern).
// ---------------------------------------------------------------------------
__global__ void exclusive_scan_kernel(int *rowPtrs, int numRows) {
    extern __shared__ int s_scan[];
    int t = threadIdx.x;

    // Load
    s_scan[t] = (t < numRows) ? rowPtrs[t] : 0;
    __syncthreads();

    // Kogge-Stone inclusive scan
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        int temp;
        if (t >= stride)
            temp = s_scan[t] + s_scan[t - stride];
        __syncthreads();
        if (t >= stride)
            s_scan[t] = temp;
        __syncthreads();
    }

    // Write — shift right by 1 for exclusive scan
    // rowPtrs[0] = 0, rowPtrs[t] = s_scan[t-1] for t>0
    if (t == 0)
        rowPtrs[0] = 0;
    else if (t < numRows)
        rowPtrs[t] = s_scan[t - 1];
    __syncthreads();

    // Last element: rowPtrs[numRows] = total nonzeros
    if (t == numRows - 1)
        rowPtrs[numRows] = s_scan[t];
}

// ---------------------------------------------------------------------------
//  Step 3: Scatter — place each nonzero into correct CSR position
//  We use atomicAdd on per-row counters in global memory to get position.
//  Alternatively, we could do this with a prefix-sum-based approach, but
//  atomics are simpler and the contention is low for many rows.
// ---------------------------------------------------------------------------
__global__ void scatter_csr_kernel(const float *coo_value,
                                   const int *coo_colIdx,
                                   const int *coo_rowIdx,
                                   const int *rowPtrs,
                                   float *csr_value, int *csr_colIdx,
                                   int *csr_rowCounter,
                                   int numNonzeros, int numRows) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numNonzeros) {
        int row = coo_rowIdx[i];
        // atomically get and increment the position within this row
        int pos = atomicAdd(&csr_rowCounter[row], 1);
        int dest = rowPtrs[row] + pos;
        csr_value[dest] = coo_value[i];
        csr_colIdx[dest] = coo_colIdx[i];
    }
}

// ---------------------------------------------------------------------------
//  CPU reference: COO to CSR conversion
// ---------------------------------------------------------------------------
void host_coo_to_csr(const float *coo_value, const int *coo_colIdx,
                     const int *coo_rowIdx, int numNonzeros, int numRows,
                     float *csr_value, int *csr_colIdx, int *rowPtrs) {
    // Count per row
    int *row_count = new int[numRows]();
    for (int i = 0; i < numNonzeros; i++)
        row_count[coo_rowIdx[i]]++;

    // Exclusive prefix sum
    rowPtrs[0] = 0;
    for (int r = 0; r < numRows; r++)
        rowPtrs[r + 1] = rowPtrs[r] + row_count[r];

    // Place elements
    int *curr_pos = new int[numRows]();
    for (int i = 0; i < numNonzeros; i++) {
        int row = coo_rowIdx[i];
        int dest = rowPtrs[row] + curr_pos[row];
        csr_value[dest] = coo_value[i];
        csr_colIdx[dest] = coo_colIdx[i];
        curr_pos[row]++;
    }

    delete[] row_count;
    delete[] curr_pos;
}

// ---------------------------------------------------------------------------
//  Main
// ---------------------------------------------------------------------------
int main() {
    CHECK_CUDA(cudaSetDevice(1));
    print_device_info(1);

    printf("╔══════════════════════════════════════════════════════════╗\n");
    printf("║  Exercise 3: COO → CSR Conversion                       ║\n");
    printf("║  Using histogram + prefix sum + scatter                  ║\n");
    printf("╚══════════════════════════════════════════════════════════╝\n\n");

    // -----------------------------------------------------------------------
    //  Test 1: Book example 4×4 matrix
    // -----------------------------------------------------------------------
    printf("--- Test 1: Book example 4×4 matrix ---\n");

    const int rows1 = 4;
    const int nnz1 = 7;

    float h_coo_val1[7]  = {1, 7, 8, 4, 3, 2, 1};
    int   h_coo_col1[7]  = {0, 2, 2, 1, 2, 0, 3};
    int   h_coo_row1[7]  = {0, 0, 1, 2, 2, 3, 3};

    // Expected CSR
    float h_csr_ref_val1[7]  = {1, 7, 8, 4, 3, 2, 1};
    int   h_csr_ref_col1[7]  = {0, 2, 2, 1, 2, 0, 3};
    int   h_csr_ref_ptr1[5]  = {0, 2, 3, 5, 7};

    // GPU conversion
    int *d_hist1, *d_rowPtrs1, *d_rowCounter1;
    float *d_csr_val1;
    int   *d_csr_col1;
    CHECK_CUDA(cudaMalloc(&d_hist1, rows1 * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_rowPtrs1, (rows1 + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_rowCounter1, rows1 * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_csr_val1, nnz1 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_csr_col1, nnz1 * sizeof(int)));

    int *d_coo_row1, *d_coo_col1;
    float *d_coo_val1;
    CHECK_CUDA(cudaMalloc(&d_coo_row1, nnz1 * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_coo_col1, nnz1 * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_coo_val1, nnz1 * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_coo_row1, h_coo_row1, nnz1 * sizeof(int),
                           cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_coo_col1, h_coo_col1, nnz1 * sizeof(int),
                           cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_coo_val1, h_coo_val1, nnz1 * sizeof(float),
                           cudaMemcpyHostToDevice));

    // Step 1: Histogram (privatized shared memory)
    int blockSize = 256;
    int gridSize  = (nnz1 + blockSize - 1) / blockSize;
    int sharedBytes = rows1 * sizeof(int);

    CHECK_CUDA(cudaMemset(d_hist1, 0, rows1 * sizeof(int)));
    histogram_row_kernel<<<gridSize, blockSize, sharedBytes>>>(
        d_coo_row1, d_hist1, nnz1, rows1);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Copy histogram → rowPtrs
    CHECK_CUDA(cudaMemcpy(d_rowPtrs1, d_hist1, rows1 * sizeof(int),
                           cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaMemset(d_rowPtrs1 + rows1, 0, sizeof(int)));

    // Step 2: Exclusive prefix sum (single block)
    exclusive_scan_kernel<<<1, 256, 256 * sizeof(int)>>>(d_rowPtrs1, rows1);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Step 3: Scatter
    CHECK_CUDA(cudaMemset(d_rowCounter1, 0, rows1 * sizeof(int)));
    scatter_csr_kernel<<<gridSize, blockSize>>>(
        d_coo_val1, d_coo_col1, d_coo_row1,
        d_rowPtrs1, d_csr_val1, d_csr_col1, d_rowCounter1, nnz1, rows1);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Read back
    float *h_csr_gpu_val1 = new float[nnz1];
    int   *h_csr_gpu_col1 = new int[nnz1];
    int   *h_csr_gpu_ptr1 = new int[rows1 + 1];
    CHECK_CUDA(cudaMemcpy(h_csr_gpu_val1, d_csr_val1, nnz1 * sizeof(float),
                           cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_csr_gpu_col1, d_csr_col1, nnz1 * sizeof(int),
                           cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_csr_gpu_ptr1, d_rowPtrs1, (rows1 + 1) * sizeof(int),
                           cudaMemcpyDeviceToHost));

    printf("  rowPtrs: [%d,%d,%d,%d,%d]  expected [%d,%d,%d,%d,%d]\n",
           h_csr_gpu_ptr1[0], h_csr_gpu_ptr1[1], h_csr_gpu_ptr1[2],
           h_csr_gpu_ptr1[3], h_csr_gpu_ptr1[4],
           h_csr_ref_ptr1[0], h_csr_ref_ptr1[1], h_csr_ref_ptr1[2],
           h_csr_ref_ptr1[3], h_csr_ref_ptr1[4]);

    bool ptr_ok = true;
    for (int i = 0; i <= rows1; i++)
        if (h_csr_gpu_ptr1[i] != h_csr_ref_ptr1[i]) ptr_ok = false;

    // Sort values within each row for comparison (scatter may reorder nonzeros)
    // Since we need rows sorted by column for exact CSR match, let's sort within each row
    for (int r = 0; r < rows1; r++) {
        int start = h_csr_gpu_ptr1[r];
        int end   = h_csr_gpu_ptr1[r + 1];
        // Bubble sort by column index within row
        for (int i = start; i < end; i++) {
            for (int j = i + 1; j < end; j++) {
                if (h_csr_gpu_col1[j] < h_csr_gpu_col1[i]) {
                    std::swap(h_csr_gpu_col1[i], h_csr_gpu_col1[j]);
                    std::swap(h_csr_gpu_val1[i], h_csr_gpu_val1[j]);
                }
            }
        }
    }

    bool val_ok = true;
    for (int i = 0; i < nnz1; i++) {
        if (std::fabs(h_csr_gpu_val1[i] - h_csr_ref_val1[i]) > 1e-5f ||
            h_csr_gpu_col1[i] != h_csr_ref_col1[i]) {
            printf("  Mismatch at index %d: value %.1f vs %.1f, col %d vs %d\n",
                   i, h_csr_gpu_val1[i], h_csr_ref_val1[i],
                   h_csr_gpu_col1[i], h_csr_ref_col1[i]);
            val_ok = false;
        }
    }

    printf("  %s\n\n", (ptr_ok && val_ok) ? "VALIDATION: PASS" : "VALIDATION: FAIL");

    delete[] h_csr_gpu_val1;
    delete[] h_csr_gpu_col1;
    delete[] h_csr_gpu_ptr1;
    CHECK_CUDA(cudaFree(d_hist1));
    CHECK_CUDA(cudaFree(d_rowPtrs1));
    CHECK_CUDA(cudaFree(d_rowCounter1));
    CHECK_CUDA(cudaFree(d_csr_val1));
    CHECK_CUDA(cudaFree(d_csr_col1));
    CHECK_CUDA(cudaFree(d_coo_row1));
    CHECK_CUDA(cudaFree(d_coo_col1));
    CHECK_CUDA(cudaFree(d_coo_val1));

    // -----------------------------------------------------------------------
    //  Test 2: Larger random COO matrix
    // -----------------------------------------------------------------------
    printf("--- Test 2: Random 1024-row COO matrix (10000 nonzeros) ---\n");

    const int rows2 = 1024;
    const int nnz2 = 10000;

    float *h_coo_val2  = new float[nnz2];
    int   *h_coo_col2  = new int[nnz2];
    int   *h_coo_row2  = new int[nnz2];
    srand(567);
    for (int i = 0; i < nnz2; i++) {
        h_coo_val2[i] = (float)(rand() % 100) / 10.0f;
        h_coo_col2[i] = rand() % 1024;
        h_coo_row2[i] = rand() % rows2;
    }

    // CPU reference
    float *h_csr_ref_val2 = new float[nnz2];
    int   *h_csr_ref_col2 = new int[nnz2];
    int   *h_csr_ref_ptr2 = new int[rows2 + 1];
    host_coo_to_csr(h_coo_val2, h_coo_col2, h_coo_row2, nnz2, rows2,
                    h_csr_ref_val2, h_csr_ref_col2, h_csr_ref_ptr2);

    // Device
    int *d_hist2, *d_rowPtrs2, *d_rowCounter2;
    float *d_csr_val2;
    int   *d_csr_col2;
    int   *d_coo_row2_d, *d_coo_col2_d;
    float *d_coo_val2_d;
    CHECK_CUDA(cudaMalloc(&d_hist2, rows2 * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_rowPtrs2, (rows2 + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_rowCounter2, rows2 * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_csr_val2, nnz2 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_csr_col2, nnz2 * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_coo_row2_d, nnz2 * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_coo_col2_d, nnz2 * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_coo_val2_d, nnz2 * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_coo_row2_d, h_coo_row2, nnz2 * sizeof(int),
                           cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_coo_col2_d, h_coo_col2, nnz2 * sizeof(int),
                           cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_coo_val2_d, h_coo_val2, nnz2 * sizeof(float),
                           cudaMemcpyHostToDevice));

    // Step 1: Histogram
    // For 1024 rows, use smaller shared mem per block: limit to 256 bins per block
    // Instead, use a coarsened histogram approach
    blockSize = 256;
    gridSize = (nnz2 + blockSize - 1) / blockSize;
    CHECK_CUDA(cudaMemset(d_hist2, 0, rows2 * sizeof(int)));
    // For rows2 = 1024, shared mem = 1024 * 4 = 4096 bytes — fine
    histogram_row_kernel<<<gridSize, blockSize, rows2 * sizeof(int)>>>(
        d_coo_row2_d, d_hist2, nnz2, rows2);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Copy histogram → rowPtrs
    CHECK_CUDA(cudaMemcpy(d_rowPtrs2, d_hist2, rows2 * sizeof(int),
                           cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaMemset(d_rowPtrs2 + rows2, 0, sizeof(int)));

    // Step 2: Exclusive prefix sum (may need multiple blocks for 1024 elements)
    // But we have max 1024 threads/block, so one block is enough
    exclusive_scan_kernel<<<1, 1024, 1024 * sizeof(int)>>>(d_rowPtrs2, rows2);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Step 3: Scatter
    CHECK_CUDA(cudaMemset(d_rowCounter2, 0, rows2 * sizeof(int)));
    scatter_csr_kernel<<<gridSize, blockSize>>>(
        d_coo_val2_d, d_coo_col2_d, d_coo_row2_d,
        d_rowPtrs2, d_csr_val2, d_csr_col2, d_rowCounter2, nnz2, rows2);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Read back
    float *h_csr_gpu_val2 = new float[nnz2];
    int   *h_csr_gpu_col2 = new int[nnz2];
    int   *h_csr_gpu_ptr2 = new int[rows2 + 1];
    CHECK_CUDA(cudaMemcpy(h_csr_gpu_val2, d_csr_val2, nnz2 * sizeof(float),
                           cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_csr_gpu_col2, d_csr_col2, nnz2 * sizeof(int),
                           cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_csr_gpu_ptr2, d_rowPtrs2, (rows2 + 1) * sizeof(int),
                           cudaMemcpyDeviceToHost));

    // Validate: compare sorted (col, value) pairs within each row
    bool val_ok2 = true;
    for (int r = 0; r < rows2; r++) {
        int start_gpu = h_csr_gpu_ptr2[r];
        int end_gpu   = h_csr_gpu_ptr2[r + 1];
        int start_ref = h_csr_ref_ptr2[r];
        int end_ref   = h_csr_ref_ptr2[r + 1];
        
        // Pair up (col, value) and sort by col then value
        std::vector<std::pair<int,float>> pairs_gpu, pairs_ref;
        for (int i = start_gpu; i < end_gpu; i++)
            pairs_gpu.push_back({h_csr_gpu_col2[i], h_csr_gpu_val2[i]});
        for (int i = start_ref; i < end_ref; i++)
            pairs_ref.push_back({h_csr_ref_col2[i], h_csr_ref_val2[i]});
        std::sort(pairs_gpu.begin(), pairs_gpu.end());
        std::sort(pairs_ref.begin(), pairs_ref.end());
        
        if (pairs_gpu.size() != pairs_ref.size()) {
            if (val_ok2) printf("  Row %d: different counts %zu vs %zu\n", r, pairs_gpu.size(), pairs_ref.size());
            val_ok2 = false;
            break;
        }
        for (size_t i = 0; i < pairs_gpu.size(); i++) {
            if (pairs_gpu[i].first != pairs_ref[i].first ||
                std::fabs(pairs_gpu[i].second - pairs_ref[i].second) > 1e-4f) {
                if (val_ok2) {
                    printf("  Row %d pair %zu: GPU (col=%d, val=%.2f) vs Ref (col=%d, val=%.2f)\n",
                           r, i,
                           pairs_gpu[i].first, pairs_gpu[i].second,
                           pairs_ref[i].first, pairs_ref[i].second);
                }
                val_ok2 = false;
            }
        }
    }

    // Validate row pointers
    bool ptr_ok2 = true;
    for (int i = 0; i <= rows2; i++) {
        if (h_csr_gpu_ptr2[i] != h_csr_ref_ptr2[i]) {
            if (ptr_ok2) {
                printf("  First rowPtr mismatch at index %d: got %d, expected %d\n",
                       i, h_csr_gpu_ptr2[i], h_csr_ref_ptr2[i]);
            }
            ptr_ok2 = false;
        }
    }

    printf("  Row pointers: %s\n", ptr_ok2 ? "OK" : "FAIL");
    printf("  Values & columns: %s\n", val_ok2 ? "OK" : "FAIL");
    printf("  %s\n\n", (ptr_ok2 && val_ok2) ? "VALIDATION: PASS" : "VALIDATION: FAIL");

    delete[] h_coo_val2;
    delete[] h_coo_col2;
    delete[] h_coo_row2;
    delete[] h_csr_ref_val2;
    delete[] h_csr_ref_col2;
    delete[] h_csr_ref_ptr2;
    delete[] h_csr_gpu_val2;
    delete[] h_csr_gpu_col2;
    delete[] h_csr_gpu_ptr2;
    CHECK_CUDA(cudaFree(d_hist2));
    CHECK_CUDA(cudaFree(d_rowPtrs2));
    CHECK_CUDA(cudaFree(d_rowCounter2));
    CHECK_CUDA(cudaFree(d_csr_val2));
    CHECK_CUDA(cudaFree(d_csr_col2));
    CHECK_CUDA(cudaFree(d_coo_row2_d));
    CHECK_CUDA(cudaFree(d_coo_col2_d));
    CHECK_CUDA(cudaFree(d_coo_val2_d));

    printf("Done.\n");
    return (ptr_ok && val_ok && ptr_ok2 && val_ok2) ? 0 : 1;
}
