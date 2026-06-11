/*
 * =============================================================================
 *  ch13_basic_radix_sort.cu — Basic Parallel Radix Sort
 * =============================================================================
 *  Book:      Programming Massively Parallel Processors (Kirk, Hwu & El Hajj,
 *             4th Edition, Morgan Kaufmann 2023)
 *  Chapter:   13 — Sorting
 *  Figures:   13.4 (parallel radix sort iteration kernel)
 *  Hardware:  GTX 1050, sm_61 (Pascal)
 *
 *  Implements:
 *    - Per-block radix sort in shared memory (Section 13.4 approach)
 *    - 16-bit LSD radix sort (1-bit radix, 16 passes)
 *    - Pairwise block merge for global sort (using Ch12 co-rank pattern)
 * =============================================================================
 */

#include "../common/cuda_utils.cuh"
#include <algorithm>
#include <cstdlib>
#include <ctime>

/* -------------------------------------------------------------------------- */
/*  Constants                                                                  */
/* -------------------------------------------------------------------------- */

#define BLOCK_SIZE 256
#define NUM_PASSES 16   // 16 bits for unsigned short

/* -------------------------------------------------------------------------- */
/*  Radix sort pass — per-block sorting in shared memory                      */
/*  Each block loads BLOCK_SIZE elements, sorts them by one bit position,     */
/*  and writes them back in coalesced order.                                  */
/* -------------------------------------------------------------------------- */

__global__ void radix_sort_pass_kernel(
    const unsigned short *input,
    unsigned short *output,
    int n,
    int pass
) {
    extern __shared__ unsigned short s_data[];
    unsigned int *s_bits = (unsigned int *)(s_data + blockDim.x);

    int t = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load key into shared memory
    unsigned short key = (i < n) ? input[i] : 0;
    s_data[t] = key;
    __syncthreads();

    // Extract bit for current pass — save before scan modifies s_bits
    unsigned int bit = (unsigned int)((key >> pass) & 1);
    s_bits[t] = bit;
    __syncthreads();

    // Inclusive scan on bits (Kogge-Stone)
    unsigned int temp;
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        __syncthreads();
        if (t >= stride)
            temp = s_bits[t] + s_bits[t - stride];
        __syncthreads();
        if (t >= stride)
            s_bits[t] = temp;
    }
    __syncthreads();

    // Total ones and zeros in this block
    unsigned int total_ones = s_bits[blockDim.x - 1];
    unsigned int total_zeros = blockDim.x - total_ones;

    // Compute destination within block
    unsigned int dest;
    if (bit == 0) {
        // Position among zeros: t - number_of_ones_before_this_thread
        // s_bits[t] is INCLUSIVE scan — count of 1s up to and including t
        dest = t - s_bits[t];
    } else {
        // Position among ones: after all zeros
        dest = total_zeros + (s_bits[t] - 1);
    }

    // Write to output (coalesced within each block)
    if (i < n)
        output[blockIdx.x * blockDim.x + dest] = s_data[t];
}

/* ========================================================================== */
/*  Co-rank function (from Ch12) — binary search for merge                    */
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

/* ========================================================================== */
/*  Sequential merge of two sorted subarrays                                  */
/* ========================================================================== */

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

/* ========================================================================== */
/*  Merge pair kernel — merge two sorted blocks using co-rank                 */
/* ========================================================================== */

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
/*  Host driver — per-block radix sort passes + merge tree                    */
/* ========================================================================== */

void sort_device(
    unsigned short *d_in,
    unsigned short *d_out,
    unsigned short *d_temp,
    int n
) {
    int numBlocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    size_t shm_size = BLOCK_SIZE * sizeof(unsigned short)
                    + BLOCK_SIZE * sizeof(unsigned int);

    // Phase 1: 16 passes of per-block radix sort
    unsigned short *cur = d_in;
    unsigned short *next = d_out;

    for (int p = 0; p < NUM_PASSES; p++) {
        radix_sort_pass_kernel<<<numBlocks, BLOCK_SIZE, shm_size>>>(
            cur, next, n, p);
        CHECK_CUDA(cudaDeviceSynchronize());
        unsigned short *tmp = cur;
        cur = next;
        next = tmp;
    }
    // After 16 passes, cur holds the per-block sorted data.
    // (Even number of swaps, so cur == d_in)

    // Phase 2: pairwise block merge until globally sorted
    int block_size = BLOCK_SIZE;
    unsigned short *buf1 = d_in;    // per-block sorted data
    unsigned short *buf2 = d_temp;

    while (block_size < n) {
        int num_pairs = (n + 2 * block_size - 1) / (2 * block_size);

        for (int pair = 0; pair < num_pairs; pair++) {
            int A_start = pair * 2 * block_size;
            int B_start = A_start + block_size;
            int m = min(block_size, n - A_start);
            int n2 = min(block_size, n - B_start);

            if (n2 <= 0) {
                // No B side — just copy A over
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

    // Final sorted result is in buf1
    if (buf1 != d_out) {
        CHECK_CUDA(cudaMemcpy(d_out, buf1, n * sizeof(unsigned short),
                              cudaMemcpyDeviceToDevice));
    }
}

/* ========================================================================== */
/*  CPU reference sort                                                        */
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

    printf("\n===== Chapter 13: Basic Radix Sort =====\n");
    printf("16-bit unsigned integers, 1-bit radix\n");
    printf("16 passes (per-block) + pairwise merge tree\n\n");

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

    printf("\n===== Basic Radix Sort Complete =====\n");
    return 0;
}
