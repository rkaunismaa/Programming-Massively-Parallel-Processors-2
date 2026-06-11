/*
 * =============================================================================
 *  ch13_merge_sort.cu — Parallel Merge Sort
 * =============================================================================
 *  Book:      Programming Massively Parallel Processors (Kirk, Hwu & El Hajj,
 *             4th Edition, Morgan Kaufmann 2023)
 *  Chapter:   13 — Sorting
 *  Section:   13.7 (Parallel merge sort)
 *  Hardware:  GTX 1050, sm_61 (Pascal)
 *
 *  Implements:
 *    - Bottom-up merge sort: sort small blocks, then merge pairs iteratively
 *    - Per-block odd-even sort in shared memory (base case)
 *    - Parallel merge using co-rank + sequential merge (from Ch12 pattern)
 *    - Comparison-based — works for any key type (here: unsigned short)
 * =============================================================================
 */

#include "../common/cuda_utils.cuh"
#include <algorithm>
#include <cstdlib>
#include <ctime>

#define BLOCK_SIZE 256

/* -------------------------------------------------------------------------- */
/*  Per-block odd-even sort in shared memory                                  */
/*  Sorts BLOCK_SIZE elements within each block using parallel odd-even       */
/*  transposition sort.                                                       */
/* -------------------------------------------------------------------------- */

__global__ void odd_even_sort_kernel(
    unsigned short *data,
    int n
) {
    extern __shared__ unsigned short s_data[];

    int t = threadIdx.x;
    int i = blockIdx.x * BLOCK_SIZE + t;

    // Load data
    s_data[t] = (i < n) ? data[i] : 65535;  // max value for padding
    __syncthreads();

    // Odd-even transposition sort on shared memory
    for (int phase = 0; phase < BLOCK_SIZE; phase++) {
        int cmp_idx;
        if (phase % 2 == 0) {
            // Even phase: compare-even (t-even with t+1)
            cmp_idx = (t % 2 == 0) ? t + 1 : -1;
        } else {
            // Odd phase: compare-odd (t-odd with t+1)
            cmp_idx = (t % 2 == 1) ? t + 1 : -1;
        }

        if (cmp_idx >= 0 && cmp_idx < BLOCK_SIZE) {
            if (s_data[t] > s_data[cmp_idx]) {
                unsigned short tmp = s_data[t];
                s_data[t] = s_data[cmp_idx];
                s_data[cmp_idx] = tmp;
            }
        }
        __syncthreads();
    }

    // Write back
    if (i < n)
        data[i] = s_data[t];
}

/* -------------------------------------------------------------------------- */
/*  Co-rank function (from Ch12)                                              */
/* -------------------------------------------------------------------------- */

__device__ __host__ int co_rank(
    int k,
    const unsigned short *A, int m,
    const unsigned short *B, int n
) {
    int i = (k < m) ? k : m;
    int j = k - i;
    int i_low = (k > n) ? (k - n) : 0;
    int j_low = (k > m) ? (k - m) : 0;
    int active = 1;
    while (active) {
        if (i > 0 && j < n && A[i - 1] > B[j]) {
            int delta = (i - i_low + 1) >> 1;
            if (delta < 1) delta = 1;
            j_low = j;   j += delta;
            i_low = i;   i -= delta;
        } else if (j > 0 && i < m && B[j - 1] > A[i]) {
            int delta = (j - j_low + 1) >> 1;
            if (delta < 1) delta = 1;
            i_low = i;   i += delta;
            j_low = j;   j -= delta;
        } else {
            active = 0;
        }
    }
    return i;
}

/* -------------------------------------------------------------------------- */
/*  Sequential merge of two sorted subarrays                                  */
/* -------------------------------------------------------------------------- */

__device__ void merge_sequential(
    const unsigned short *A, int m,
    const unsigned short *B, int n,
    unsigned short *C
) {
    int i = 0, j = 0, k = 0;
    while (i < m && j < n) {
        if (A[i] <= B[j]) C[k++] = A[i++];
        else              C[k++] = B[j++];
    }
    while (i < m) C[k++] = A[i++];
    while (j < n) C[k++] = B[j++];
}

/* -------------------------------------------------------------------------- */
/*  Merge pair kernel — merge two sorted blocks using co-rank                 */
/* -------------------------------------------------------------------------- */

__global__ void merge_pair_kernel(
    const unsigned short *A, int m,
    const unsigned short *B, int n,
    unsigned short *C
) {
    int total = m + n;
    int total_threads = gridDim.x * blockDim.x;
    int elems_per_thread = (total + total_threads - 1) / total_threads;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int k_curr = tid * elems_per_thread;
    int k_next = min((tid + 1) * elems_per_thread, total);
    if (k_curr >= total) return;
    int i_curr = co_rank(k_curr, A, m, B, n);
    int j_curr = k_curr - i_curr;
    int i_next = co_rank(k_next, A, m, B, n);
    int j_next = k_next - i_next;
    merge_sequential(&A[i_curr], i_next - i_curr,
                     &B[j_curr], j_next - j_curr,
                     &C[k_curr]);
}

/* ========================================================================== */
/*  Host driver — sort small blocks, then merge tree                          */
/* ========================================================================== */

void merge_sort_device(
    unsigned short *d_in,
    unsigned short *d_out,
    unsigned short *d_temp,
    int n
) {
    // Phase 1: Sort each 256-element block using odd-even sort
    int numBlocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    size_t shm_size = BLOCK_SIZE * sizeof(unsigned short);

    odd_even_sort_kernel<<<numBlocks, BLOCK_SIZE, shm_size>>>(d_in, n);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Phase 2: Iterative pairwise merge
    int block_size = BLOCK_SIZE;
    unsigned short *buf1 = d_in;
    unsigned short *buf2 = d_temp;

    while (block_size < n) {
        int num_pairs = (n + 2 * block_size - 1) / (2 * block_size);

        for (int pair = 0; pair < num_pairs; pair++) {
            int A_start = pair * 2 * block_size;
            int B_start = A_start + block_size;
            int m = min(block_size, n - A_start);
            int n2 = min(block_size, n - B_start);

            if (n2 <= 0) {
                if (A_start < n) {
                    CHECK_CUDA(cudaMemcpy(&buf2[A_start], &buf1[A_start],
                                          m * sizeof(unsigned short),
                                          cudaMemcpyDeviceToDevice));
                }
                continue;
            }

            int total = m + n2;
            int threads = 256;
            int g_blocks = (total + threads * 8 - 1) / (threads * 8);

            merge_pair_kernel<<<g_blocks, threads>>>(
                &buf1[A_start], m,
                &buf1[B_start], n2,
                &buf2[A_start]);
            CHECK_CUDA(cudaDeviceSynchronize());
        }

        block_size *= 2;
        unsigned short *tmp = buf1;
        buf1 = buf2;
        buf2 = tmp;
    }

    if (buf1 != d_out) {
        CHECK_CUDA(cudaMemcpy(d_out, buf1, n * sizeof(unsigned short),
                              cudaMemcpyDeviceToDevice));
    }
}

/* ========================================================================== */
/*  CPU reference                                                             */
/* ========================================================================== */

void sort_cpu(unsigned short *data, int n) {
    std::sort(data, data + n);
}

/* ========================================================================== */
/*  Main                                                                      */
/* ========================================================================== */

int main() {
    int dev_id = 1;
    cudaSetDevice(dev_id);
    print_device_info(dev_id);

    srand(time(nullptr));

    printf("\n===== Chapter 13: Parallel Merge Sort =====\n");
    printf("16-bit unsigned integers, odd-even per-block sort + merge tree\n\n");

    int test_sizes[] = {256, 1024, 4096, 16384, 65536, 262144, 524288, 1048576};
    int num_tests = sizeof(test_sizes) / sizeof(test_sizes[0]);

    for (int t = 0; t < num_tests; t++) {
        int n = test_sizes[t];

        unsigned short *h_keys = new unsigned short[n];
        unsigned short *h_ref = new unsigned short[n];

        for (int i = 0; i < n; i++)
            h_keys[i] = (unsigned short)(rand() & 0xFFFF);

        memcpy(h_ref, h_keys, n * sizeof(unsigned short));
        sort_cpu(h_ref, n);

        unsigned short *d_in, *d_out, *d_temp;
        CHECK_CUDA(cudaMalloc(&d_in,   n * sizeof(unsigned short)));
        CHECK_CUDA(cudaMalloc(&d_out,  n * sizeof(unsigned short)));
        CHECK_CUDA(cudaMalloc(&d_temp, n * sizeof(unsigned short)));

        CHECK_CUDA(cudaMemcpy(d_in, h_keys, n * sizeof(unsigned short),
                              cudaMemcpyHostToDevice));

        gpu_timer timer;
        timer.start();
        merge_sort_device(d_in, d_out, d_temp, n);
        timer.stop();
        float ms = timer.elapsed_ms();

        unsigned short *h_result = new unsigned short[n];
        CHECK_CUDA(cudaMemcpy(h_result, d_out, n * sizeof(unsigned short),
                              cudaMemcpyDeviceToHost));

        bool sorted = true;
        bool matches = true;
        for (int i = 0; i < n; i++) {
            if (i > 0 && h_result[i] < h_result[i - 1]) {
                printf("  NOT SORTED at index %d: %u < %u\n",
                       i, h_result[i], h_result[i - 1]);
                sorted = false;
                break;
            }
        }
        if (sorted) {
            for (int i = 0; i < n; i++) {
                if (h_result[i] != h_ref[i]) {
                    printf("  VALUE MISMATCH at index %d: got %u, expected %u\n",
                           i, h_result[i], h_ref[i]);
                    matches = false;
                    break;
                }
            }
        }

        float throughput = (n * sizeof(unsigned short) * 2.0f) / (ms * 1e6f);
        printf("  n=%7d | %s | %7.2f ms | %5.2f GB/s\n",
               n,
               (sorted && matches) ? "PASS" : "FAIL",
               ms, throughput);

        CHECK_CUDA(cudaFree(d_in));
        CHECK_CUDA(cudaFree(d_out));
        CHECK_CUDA(cudaFree(d_temp));
        delete[] h_keys;
        delete[] h_ref;
        delete[] h_result;
    }

    printf("\n===== Parallel Merge Sort Complete =====\n");
    return 0;
}
