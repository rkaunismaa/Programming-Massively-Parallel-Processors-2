/*
 * =============================================================================
 *  ch12_circular_buffer_merge.cu — Circular Buffer Merge Kernel
 * =============================================================================
 *  Book:      Programming Massively Parallel Processors (Kirk, Hwu & El Hajj,
 *             4th Edition, Morgan Kaufmann 2023)
 *  Chapter:   12 — Merge
 *  Figures:   12.16, 12.18, 12.19, 12.20
 *  Hardware:  GTX 1050, sm_61 (Pascal)
 *
 *  Key improvement over tiled merge: circular buffer management of shared memory
 *  reuses unconsumed elements from previous iterations — no data is wasted.
 *  Each iteration refills only the portion of the buffer consumed, not the full tile.
 *
 *  co_rank_circular() and merge_sequential_circular() expose a "simplified model"
 *  to calling code: co-rank values are linear offsets (0..tile_size) even though
 *  the underlying shared memory is a circular buffer. Internal index translation
 *  handles wrapping via cir_idx().
 *
 *  Critical pitfalls addressed:
 *   - delta=0 infinite loop in co_rank binary search (both co_rank and co_rank_circular)
 *   - Negative circular index wrap in cir_idx(i-1, ...)
 *   - Register variables updated by thread 0 must be broadcast through shared memory
 *   - tile_out capped by available elements (A_avail + B_avail)
 *   - Buffer occupancy recomputed with correct formula after refill
 *   - Thread-level calls pass full buffer pointers with adjusted circular starts
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
/*  Circular Buffer Index Helper                                              */
/*  Converts a linear offset (relative to circular start) into a physical      */
/*  shared memory index, wrapping as needed. Handles negative offsets.         */
/* -------------------------------------------------------------------------- */

__device__ __host__ inline int cir_idx(int linear_idx, int start, int tile_size) {
    int idx = start + linear_idx;
    if (idx < 0) idx += tile_size;         // CRITICAL: handle negative wrap
    if (idx >= tile_size) idx -= tile_size;
    return idx;
}

/* -------------------------------------------------------------------------- */
/*  Fig 12.19 — co_rank_circular: co-rank on circular buffer                  */
/* -------------------------------------------------------------------------- */

__device__ int co_rank_circular(
    int k,
    const int *A_s, int m,
    const int *B_s, int n,
    int A_S_start, int B_S_start, int tile_size
) {
    int i = (k < m) ? k : m;
    int j = k - i;
    int i_low = (k > n) ? (k - n) : 0;
    int j_low = (k > m) ? (k - m) : 0;
    int active = 1;
    while (active) {
        // Convert linear offsets to circular buffer indices
        int i_m1_cir = cir_idx(i - 1, A_S_start, tile_size);
        int i_cir     = cir_idx(i,     A_S_start, tile_size);
        int j_cir     = cir_idx(j,     B_S_start, tile_size);
        int j_m1_cir = cir_idx(j - 1, B_S_start, tile_size);

        if (i > 0 && j < n && A_s[i_m1_cir] > B_s[j_cir]) {
            int delta = (i - i_low + 1) >> 1;
            if (delta < 1) delta = 1;
            j_low = j;  j += delta;
            i_low = i;  i -= delta;
        } else if (j > 0 && i < m && B_s[j_m1_cir] > A_s[i_cir]) {
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
/*  Fig 12.20 — merge_sequential_circular: merge on circular buffer            */
/* -------------------------------------------------------------------------- */

__device__ void merge_sequential_circular(
    const int *A_s, int m,
    const int *B_s, int n,
    int *C,
    int A_S_start, int B_S_start, int tile_size
) {
    int i = 0, j = 0, k = 0;
    while (i < m && j < n) {
        int ao = cir_idx(i, A_S_start, tile_size);
        int bo = cir_idx(j, B_S_start, tile_size);
        if (A_s[ao] <= B_s[bo]) {
            C[k++] = A_s[ao];
            i++;
        } else {
            C[k++] = B_s[bo];
            j++;
        }
    }
    while (i < m) {
        C[k++] = A_s[cir_idx(i, A_S_start, tile_size)];
        i++;
    }
    while (j < n) {
        C[k++] = B_s[cir_idx(j, B_S_start, tile_size)];
        j++;
    }
}

/* -------------------------------------------------------------------------- */
/*  Circular Buffer Merge Kernel (Fig 12.16 + 12.18 + 12.19 + 12.20)          */
/* -------------------------------------------------------------------------- */

#define TILE_SIZE 1024

__global__ void merge_circular_buffer_kernel(
    const int *A, int m,
    const int *B, int n,
    int *C
) {
    __shared__ int A_s[TILE_SIZE];
    __shared__ int B_s[TILE_SIZE];
    __shared__ int s_bc[8];  // Dedicated broadcast array (avoids overwriting buffer data)

    int total = m + n;
    int num_blocks = gridDim.x;
    int block_id = blockIdx.x;
    int tid = threadIdx.x;

    // ---- Block-level output partition (Fig 12.11) ----
    int elems_per_block = (total + num_blocks - 1) / num_blocks;
    int C_curr = block_id * elems_per_block;
    int C_next = min((block_id + 1) * elems_per_block, total);
    int C_length = C_next - C_curr;

    // Block-level co-rank (thread 0 only on global memory)
    int A_curr;
    if (tid == 0) {
        A_s[0] = co_rank(C_curr, A, m, B, n);
        A_s[1] = co_rank(C_next, A, m, B, n);
    }
    __syncthreads();

    A_curr = A_s[0];
    int A_next = A_s[1];
    int block_B_curr = C_curr - A_curr;
    int block_B_next = C_next - A_next;
    int A_length = A_next - A_curr;
    int B_length = block_B_next - block_B_curr;

    // ---- Circular buffer state ----
    int A_S_start = 0;
    int B_S_start = 0;
    int A_consumed = 0;
    int B_consumed = 0;
    int C_completed = 0;

    // Load initial tiles
    int Aload_init = min(TILE_SIZE, A_length);
    int Bload_init = min(TILE_SIZE, B_length);
    int Aload = Aload_init;
    int Bload = Bload_init;
    for (int i = tid; i < Aload; i += blockDim.x) {
        if (i < A_length) {
            A_s[i] = A[A_curr + i];
        }
    }
    for (int i = tid; i < Bload; i += blockDim.x) {
        if (i < B_length) {
            B_s[i] = B[block_B_curr + i];
        }
    }
    __syncthreads();

    while (C_completed < C_length) {
        int A_avail = Aload;
        int B_avail = Bload;
        int tile_out = min(TILE_SIZE, C_length - C_completed);
        tile_out = min(tile_out, A_avail + B_avail);  // CRITICAL: cap by available

        // ---- Fig 12.18: Thread-level merge from circular buffer ----
        int threads_in_block = blockDim.x;
        int elems_per_thread = (tile_out + threads_in_block - 1) / threads_in_block;
        int k_curr = tid * elems_per_thread;
        int k_next = min((tid + 1) * elems_per_thread, tile_out);

        if (k_curr < tile_out) {
            int i_curr = co_rank_circular(k_curr, A_s, A_avail, B_s, B_avail,
                                          A_S_start, B_S_start, TILE_SIZE);
            int i_next = co_rank_circular(k_next, A_s, A_avail, B_s, B_avail,
                                          A_S_start, B_S_start, TILE_SIZE);

            int len_a = i_next - i_curr;
            int len_b = k_next - k_curr - len_a;
            int j_curr = k_curr - i_curr;

            // Pass full buffer pointers; adjust circular starts by subarray offset
            int adj_ASS = A_S_start + i_curr;
            if (adj_ASS >= TILE_SIZE) adj_ASS -= TILE_SIZE;
            int adj_BSS = B_S_start + j_curr;
            if (adj_BSS >= TILE_SIZE) adj_BSS -= TILE_SIZE;
            merge_sequential_circular(A_s, len_a, B_s, len_b,
                                       &C[C_curr + C_completed + k_curr],
                                       adj_ASS, adj_BSS, TILE_SIZE);
        }
        __syncthreads();

        // ---- Update consumption tracking (thread 0) ----
        int A_used_this_iter = 0;
        if (tid == 0) {
            int total_A_used = co_rank_circular(tile_out, A_s, A_avail, B_s, B_avail,
                                                A_S_start, B_S_start, TILE_SIZE);
            A_consumed += total_A_used;
            B_consumed += (tile_out - total_A_used);
            C_completed += tile_out;

            // Update circular buffer start positions
            A_S_start = cir_idx(total_A_used, A_S_start, TILE_SIZE);
            B_S_start = cir_idx(tile_out - total_A_used, B_S_start, TILE_SIZE);

            // Broadcast all counters through shared memory
            s_bc[0] = A_consumed;
            s_bc[1] = B_consumed;
            s_bc[2] = C_completed;
            s_bc[3] = A_S_start;
            s_bc[4] = B_S_start;
            s_bc[5] = total_A_used;   // A_used_this_iter
        }
        __syncthreads();

        A_consumed   = s_bc[0];
        B_consumed   = s_bc[1];
        C_completed  = s_bc[2];
        A_S_start    = s_bc[3];
        B_S_start    = s_bc[4];
        A_used_this_iter = s_bc[5];

        // ---- Fig 12.16: Refill consumed portion from global memory ----
        if (C_completed < C_length) {
            // Compute B used this iteration
            int B_used_this_iter = tile_out - A_used_this_iter;

            // Refill A: load from AFTER all elements loaded into buffer so far
            // gi = A_curr + Acon + A_avail - Au  (where Acon = updated consumption, A_avail = old buffer occupancy)
            int gi_A = A_curr + A_consumed + A_avail - A_used_this_iter;
            int rem_A = max(0, A_length - A_consumed - A_avail + A_used_this_iter);
            for (int i = tid; i < A_used_this_iter; i += blockDim.x) {
                int buffer_idx = cir_idx(A_avail - A_used_this_iter + i,
                                         A_S_start, TILE_SIZE);
                if (i < rem_A) {
                    A_s[buffer_idx] = A[gi_A + i];
                }
            }

            // Refill B: same pattern
            int gi_B = block_B_curr + B_consumed + B_avail - B_used_this_iter;
            int rem_B = max(0, B_length - B_consumed - B_avail + B_used_this_iter);
            for (int i = tid; i < B_used_this_iter; i += blockDim.x) {
                int buffer_idx = cir_idx(B_avail - B_used_this_iter + i,
                                         B_S_start, TILE_SIZE);
                if (i < rem_B) {
                    B_s[buffer_idx] = B[gi_B + i];
                }
            }
            __syncthreads();

            // Recompute actual buffer occupancy
            Aload = A_avail - A_used_this_iter + min(A_used_this_iter, rem_A);
            Bload = B_avail - B_used_this_iter + min(B_used_this_iter, rem_B);
        }
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

    printf("\n===== ch12_circular_buffer_merge — Circular Buffer Merge (Tile=%d) =====\n", TILE_SIZE);
    printf("Circular buffer reuses unconsumed elements — no reloading waste\n");
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
            merge_circular_buffer_kernel<<<blocks, threads>>>(d_A, m, d_B, n, d_C);
            CHECK_CUDA(cudaDeviceSynchronize());

            gpu_timer timer;
            timer.start();
            merge_circular_buffer_kernel<<<blocks, threads>>>(d_A, m, d_B, n, d_C);
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

    printf("===== Circular Buffer Merge Kernel Complete =====\n");
    return 0;
}
