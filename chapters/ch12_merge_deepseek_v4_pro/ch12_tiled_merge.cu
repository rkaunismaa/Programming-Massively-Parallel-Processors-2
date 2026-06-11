/*
 * =============================================================================
 *  ch12_tiled_merge.cu — Tiled Parallel Merge Kernel (Shared Memory)
 * =============================================================================
 *  Book:      Programming Massively Parallel Processors (Kirk, Hwu & El Hajj,
 *             4th Edition, Morgan Kaufmann 2023)
 *  Chapter:   12 — Merge
 *  Figures:   12.11, 12.12, 12.13
 *  Hardware:  GTX 1050, sm_61 (Pascal)
 *
 *  Key improvements over basic merge:
 *    1. Block-level co-rank: one thread per block calls co_rank on global memory
 *       (instead of every thread). Reduces global binary searches from #threads to #blocks.
 *    2. Cooperative tile loading: all threads load A and B tiles into shared memory
 *       in coalesced patterns.
 *    3. Thread-level co-rank on SHARED memory: individual threads call co_rank
 *       on the in-shared-memory tiles (cheaper than global memory co-rank).
 *
 *  Limitation: ~50% of loaded data consumed per iteration; rest reloaded next time.
 *  The circular buffer kernel (ch12_circular_buffer_merge.cu) addresses this.
 *
 *  Counter broadcast: Variables updated by thread 0 (A_consumed, B_consumed,
 *  C_completed) are broadcast through shared memory with __syncthreads() fences
 *  so all threads see the updated values before the next while-loop iteration.
 * =============================================================================
 */

#include "../common/cuda_utils.cuh"
#include <algorithm>
#include <cstdlib>
#include <ctime>

/* -------------------------------------------------------------------------- */
/*  Fig 12.2 — Sequential Merge (host + device)                               */
/* -------------------------------------------------------------------------- */

__device__ __host__ void merge_sequential(
    const int *A, int m,
    const int *B, int n,
    int *C
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
/*  Fig 12.5 — Co-rank Function (host + device)                               */
/* -------------------------------------------------------------------------- */

__device__ __host__ int co_rank(
    int k,
    const int *A, int m,
    const int *B, int n
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
            j_low = j;  j += delta;
            i_low = i;  i -= delta;
        } else if (j > 0 && i < m && B[j - 1] > A[i]) {
            int delta = (j - j_low + 1) >> 1;
            if (delta < 1) delta = 1;
            i_low = i;  i += delta;
            j_low = j;  j -= delta;
        } else {
            active = 0;
        }
    }
    return i;
}

/* -------------------------------------------------------------------------- */
/*  Tiled Merge Kernel                                                        */
/* -------------------------------------------------------------------------- */

#define TILE_SIZE 1024

__global__ void merge_tiled_kernel(
    const int *A, int m,
    const int *B, int n,
    int *C
) {
    __shared__ int A_s[TILE_SIZE];
    __shared__ int B_s[TILE_SIZE];

    int total = m + n;
    int num_blocks = gridDim.x;
    int block_id = blockIdx.x;
    int tid = threadIdx.x;

    // ---- Fig 12.11: Block-level partition ----
    int elems_per_block = (total + num_blocks - 1) / num_blocks;
    int C_curr = block_id * elems_per_block;
    int C_next = min((block_id + 1) * elems_per_block, total);
    int C_length = C_next - C_curr;

    // Block-level co-rank: only thread 0 calls on global memory
    if (tid == 0) {
        A_s[0] = co_rank(C_curr, A, m, B, n);  // block-level A start
        A_s[1] = co_rank(C_next, A, m, B, n);  // block-level A end
    }
    __syncthreads();

    int A_curr = A_s[0];
    int A_next = A_s[1];
    int B_curr = C_curr - A_curr;
    int B_next = C_next - A_next;
    int A_length = A_next - A_curr;
    int B_length = B_next - B_curr;

    // Iterative tile processing
    int A_consumed = 0;
    int B_consumed = 0;
    int C_completed = 0;

    while (C_completed < C_length) {
        int tile_out = min(TILE_SIZE, C_length - C_completed);

        // ---- Fig 12.12: Load A tile into shared memory ----
        int A_remaining = A_length - A_consumed;
        int A_to_load = min(TILE_SIZE, A_remaining);
        for (int i = tid; i < A_to_load; i += blockDim.x) {
            if (i < A_remaining) {
                A_s[i] = A[A_curr + A_consumed + i];
            }
        }

        // ---- Load B tile into shared memory ----
        int B_remaining = B_length - B_consumed;
        int B_to_load = min(TILE_SIZE, B_remaining);
        for (int i = tid; i < B_to_load; i += blockDim.x) {
            if (i < B_remaining) {
                B_s[i] = B[B_curr + B_consumed + i];
            }
        }
        __syncthreads();

        // ---- Fig 12.13: Thread-level merge from shared memory ----
        int threads_in_block = blockDim.x;
        int elems_per_thread = (tile_out + threads_in_block - 1) / threads_in_block;
        int k_curr = tid * elems_per_thread;
        int k_next = min((tid + 1) * elems_per_thread, tile_out);

        if (k_curr < tile_out) {
            // Co-rank on SHARED memory data (A_s, B_s), not global (A, B)
            int i_curr = co_rank(k_curr, A_s, A_to_load, B_s, B_to_load);
            int i_next = co_rank(k_next, A_s, A_to_load, B_s, B_to_load);

            int len_a = i_next - i_curr;
            int len_b = k_next - k_curr - len_a;
            int j_curr = k_curr - i_curr;

            // Merge from shared memory into global output
            merge_sequential(&A_s[i_curr], len_a,
                             &B_s[j_curr], len_b,
                             &C[C_curr + C_completed + k_curr]);
        }
        __syncthreads();

        // Update consumption counters (thread 0 only, then broadcast)
        if (tid == 0) {
            int total_A_used = co_rank(tile_out, A_s, A_to_load, B_s, B_to_load);
            A_consumed += total_A_used;
            B_consumed += (tile_out - total_A_used);
            C_completed += tile_out;

            // Broadcast through shared memory (register-to-shared + __syncthreads fence)
            A_s[0] = A_consumed;
            B_s[0] = B_consumed;
            A_s[2] = C_completed;
        }
        __syncthreads();

        A_consumed  = A_s[0];
        B_consumed  = B_s[0];
        C_completed = A_s[2];
    }
}

/* -------------------------------------------------------------------------- */
/*  CPU Reference                                                              */
/* -------------------------------------------------------------------------- */

void host_merge(const int *A, int m, const int *B, int n, int *C) {
    merge_sequential(A, m, B, n, C);
}

/* -------------------------------------------------------------------------- */
/*  Helper — generate a sorted array of unique values                          */
/* -------------------------------------------------------------------------- */

void generate_sorted_array(int *arr, int size, int max_val) {
    for (int i = 0; i < size; i++) {
        arr[i] = rand() % max_val;
    }
    std::sort(arr, arr + size);
    if (size > 0) {
        int *end = std::unique(arr, arr + size);
        int new_size = (int)(end - arr);
        for (int i = new_size; i < size; i++) {
            arr[i] = max_val + i - new_size;
        }
    }
}

/* -------------------------------------------------------------------------- */
/*  Main                                                                      */
/* -------------------------------------------------------------------------- */

int main() {
    int dev_id = 1;
    cudaSetDevice(dev_id);
    print_device_info(dev_id);

    int test_sizes[] = {128, 1024, 4096, 32768, 131072, 524288};
    int num_tests = sizeof(test_sizes) / sizeof(test_sizes[0]);
    srand(time(nullptr));

    printf("\n===== ch12_tiled_merge — Tiled Merge (Tile=%d) =====\n", TILE_SIZE);
    printf("Block-level co-rank + cooperative tile load + shared-memory merge\n");
    printf("Testing 3 A:B size ratios per total size (20%% diff, 50%% diff, 80%% diff)\n\n");

    for (int t = 0; t < num_tests; t++) {
        int total = test_sizes[t];

        for (int ratio = 0; ratio < 3; ratio++) {
            int m, n;
            if (ratio == 0)      { m = total / 5;     n = total - m; }
            else if (ratio == 1) { m = total / 2;     n = total - m; }
            else                 { m = 4 * total / 5;  n = total - m; }

            int max_val = total * 4;

            int *h_A = new int[m];
            int *h_B = new int[n];
            int *h_C_ref = new int[total];
            int *h_C_gpu = new int[total];

            generate_sorted_array(h_A, m, max_val);
            generate_sorted_array(h_B, n, max_val);
            host_merge(h_A, m, h_B, n, h_C_ref);

            int *d_A, *d_B, *d_C;
            CHECK_CUDA(cudaMalloc(&d_A, m * sizeof(int)));
            CHECK_CUDA(cudaMalloc(&d_B, n * sizeof(int)));
            CHECK_CUDA(cudaMalloc(&d_C, total * sizeof(int)));

            CHECK_CUDA(cudaMemcpy(d_A, h_A, m * sizeof(int), cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMemcpy(d_B, h_B, n * sizeof(int), cudaMemcpyHostToDevice));

            int threads = 128;
            int blocks = 16;

            // Warm-up
            merge_tiled_kernel<<<blocks, threads>>>(d_A, m, d_B, n, d_C);
            CHECK_CUDA(cudaDeviceSynchronize());

            gpu_timer timer;
            timer.start();
            merge_tiled_kernel<<<blocks, threads>>>(d_A, m, d_B, n, d_C);
            timer.stop();
            float ms = timer.elapsed_ms();

            CHECK_CUDA(cudaMemcpy(h_C_gpu, d_C, total * sizeof(int), cudaMemcpyDeviceToHost));

            // Validate (int comparison — exact match required)
            bool pass = true;
            for (int i = 0; i < total; i++) {
                if (h_C_ref[i] != h_C_gpu[i]) {
                    printf("  FAIL: first mismatch at index %d: expected %d, got %d\n",
                           i, h_C_ref[i], h_C_gpu[i]);
                    pass = false;
                    break;
                }
            }

            float throughput = (total * sizeof(int) * 3.0f) / (ms * 1e6f);
            printf("  N=%6d (m=%5d,n=%5d) | %s | %7.3f ms | %6.2f GB/s\n",
                   total, m, n, pass ? "PASS" : "FAIL", ms, throughput);

            CHECK_CUDA(cudaFree(d_A));
            CHECK_CUDA(cudaFree(d_B));
            CHECK_CUDA(cudaFree(d_C));
            delete[] h_A;
            delete[] h_B;
            delete[] h_C_ref;
            delete[] h_C_gpu;
        }
        printf("\n");
    }

    printf("===== Tiled Merge Kernel Complete =====\n");
    return 0;
}
