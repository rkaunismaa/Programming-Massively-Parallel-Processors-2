/*
 * =============================================================================
 *  ch13_tiled_radix_sort.cu — Tiled Radix Sort (4-bit radix, 16 buckets)
 * =============================================================================
 *  Book:      Programming Massively Parallel Processors (Kirk, Hwu & El Hajj,
 *             4th Edition, Morgan Kaufmann 2023)
 *  Chapter:   13 — Sorting
 *  Hardware:  GTX 1050, sm_61 (Pascal)
 *
 *  Uses scan-based stable bucket placement within each block.
 *  Each pass processes 4 bits: for each of 16 digit values, computes
 *  an inclusive scan on a per-digit flag, then writes keys to their
 *  correct position within the digit group.
 *
 *  4-bit LSD radix sort: 4 passes for 16-bit keys.
 * =============================================================================
 */

#include "../common/cuda_utils.cuh"
#include <algorithm>
#include <cstdlib>
#include <ctime>

#define BLOCK_SIZE 256
#define RADIX 16
#define NUM_PASSES 4

/*
 * Per-block radix-16 pass using scan-based stable bucket placement.
 *
 * The challenge with 16 buckets is determining each key's position
 * within its digit group while maintaining stability (preserving input
 * order for equal-digit keys).
 *
 * Approach: For each digit value d (0..15):
 *   1. Set s_flag[t] = 1 if this thread's digit == d, else 0
 *   2. Inclusive scan on s_flag
 *   3. For thread with digit == d:
 *        position_within_digit = s_flag[t] - 1
 *        dest = base_offset[d] + position_within_digit
 *        output[block_start + dest] = s_keys[t]
 *
 * The scan assigns positions in thread-index order (matching input order),
 * making the sort stable.
 */

__global__ void radix16_pass_kernel(
    const unsigned short *input,
    unsigned short *output,
    int n,
    int pass
) {
    extern __shared__ unsigned int shared[];

    int t = threadIdx.x;
    int i = blockIdx.x * BLOCK_SIZE + t;
    int block_start = blockIdx.x * BLOCK_SIZE;

    // Shared memory layout (unsigned int):
    //   [0 .. BLOCK_SIZE-1]     : s_digit
    //   [BLOCK_SIZE .. BLOCK_SIZE+RADIX-1] : s_hist
    unsigned int *s_digit = shared;
    int *s_hist = (int *)(shared + BLOCK_SIZE);

    // Load key and extract 4-bit digit
    unsigned short key = (i < n) ? input[i] : 0;
    unsigned int d = (unsigned int)((key >> (pass * 4)) & 0xF);
    s_digit[t] = d;
    __syncthreads();

    // Build histogram: thread coarsening for counting
    if (t < RADIX) {
        int count = 0;
        for (int j = 0; j < BLOCK_SIZE; j++)
            if (s_digit[j] == (unsigned int)t) count++;
        s_hist[t] = count;
    }
    __syncthreads();

    // Exclusive prefix sum on histogram (thread 0)
    __shared__ int s_offset[RADIX];
    if (t == 0) {
        int sum = 0;
        for (int b = 0; b < RADIX; b++) {
            s_offset[b] = sum;
            sum += s_hist[b];
        }
    }
    __syncthreads();

    // Process each digit value with stable scan-based placement
    // For each thread, compute position within its digit group by
    // counting same-digit keys before it in input order.
    // This is O(BLOCK_SIZE) per thread but deterministic and stable.
    int pos_in_digit = 0;
    for (int j = 0; j < t; j++) {
        if (s_digit[j] == d) pos_in_digit++;
    }
    int dest = s_offset[d] + pos_in_digit;
    if (i < n)
        output[block_start + dest] = key;
}

/* ========================================================================== */
/*  Co-rank + merge utilities (from Ch12)                                     */
/* ========================================================================== */

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
/*  Host driver                                                               */
/* ========================================================================== */

void sort_device(
    unsigned short *d_in,
    unsigned short *d_out,
    unsigned short *d_temp,
    int n
) {
    int numBlocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Shared memory: s_digit (unsigned int) + s_hist (int)
    // s_digit[0..BLOCK_SIZE-1], then s_hist[0..RADIX-1]
    size_t shm_size = (BLOCK_SIZE + RADIX) * sizeof(unsigned int);

    // Phase 1: 4 passes of radix-16 sort
    unsigned short *cur = d_in;
    unsigned short *next = d_out;

    for (int p = 0; p < NUM_PASSES; p++) {
        radix16_pass_kernel<<<numBlocks, BLOCK_SIZE, shm_size>>>(
            cur, next, n, p);
        CHECK_CUDA(cudaDeviceSynchronize());
        unsigned short *tmp = cur;
        cur = next;
        next = tmp;
    }

    // Phase 2: pairwise block merge
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

    printf("\n===== Chapter 13: Tiled Radix Sort (4-bit radix) =====\n");
    printf("16-bit unsigned integers, 4-bit radix (16 buckets)\n");
    printf("4 passes (scan-based stable per-block sort) + pairwise merge\n\n");

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
        sort_device(d_in, d_out, d_temp, n);
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

        float throughput = (n * sizeof(unsigned short) * 3.0f) / (ms * 1e6f);
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

    printf("\n===== Tiled Radix Sort Complete =====\n");
    return 0;
}
