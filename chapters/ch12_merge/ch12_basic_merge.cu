/*
 * =============================================================================
 *  ch12_basic_merge.cu — Basic Parallel Merge Kernel
 * =============================================================================
 *  Book:      Programming Massively Parallel Processors (Kirk, Hwu & El Hajj,
 *             4th Edition, Morgan Kaufmann 2023)
 *  Chapter:   12 — Merge
 *  Figures:   12.2, 12.5, 12.9
 *  Hardware:  GTX 1050, sm_61 (Pascal)
 *
 *  Implements:
 *    - Fig 12.2:    merge_sequential() — sequential merge of two sorted subarrays
 *    - Fig 12.5:    co_rank() — binary-search-based co-rank function
 *    - Fig 12.9:    merge_basic_kernel() — basic parallel merge, one thread per
 *                   output segment; each thread calls co_rank to determine its
 *                   input ranges, then merge_sequential on its subarrays.
 * =============================================================================
 */

#include "../common/cuda_utils.cuh"
#include <algorithm>
#include <cstdlib>
#include <ctime>

/* -------------------------------------------------------------------------- */
/*  Fig 12.2 — Sequential Merge Function                                      */
/* -------------------------------------------------------------------------- */

__device__ __host__ void merge_sequential(
    const int *A, int m,
    const int *B, int n,
    int *C
) {
    int i = 0, j = 0, k = 0;
    while (i < m && j < n) {
        if (A[i] <= B[j]) {
            C[k++] = A[i++];
        } else {
            C[k++] = B[j++];
        }
    }
    while (i < m) C[k++] = A[i++];
    while (j < n) C[k++] = B[j++];
}

/* -------------------------------------------------------------------------- */
/*  Fig 12.5 — Co-rank Function (binary search)                               */
/* -------------------------------------------------------------------------- */

__device__ __host__ int co_rank(
    int k,              // rank (output position) in the merged array
    const int *A, int m,  // sorted input array A, size m
    const int *B, int n   // sorted input array B, size n
) {
    // i = number of A elements in the first k elements of C
    int i = (k < m) ? k : m;
    int j = k - i;

    // Lower bounds: at least (k - n) from A and (k - m) from B
    int i_low = (k > n) ? (k - n) : 0;
    int j_low = (k > m) ? (k - m) : 0;

    int active = 1;
    while (active) {
        if (i > 0 && j < n && A[i - 1] > B[j]) {
            // i is too high — more A elements than necessary
            int delta = (i - i_low + 1) >> 1;
            if (delta < 1) delta = 1;
            j_low = j;
            j += delta;
            i_low = i;
            i -= delta;
        } else if (j > 0 && i < m && B[j - 1] > A[i]) {
            // j is too high — more B elements than necessary
            int delta = (j - j_low + 1) >> 1;
            if (delta < 1) delta = 1;
            i_low = i;
            i += delta;
            j_low = j;
            j -= delta;
        } else {
            active = 0;
        }
    }
    return i;  // co-rank: number of A elements in the first k output elements
}

/* -------------------------------------------------------------------------- */
/*  Fig 12.9 — Basic Parallel Merge Kernel                                    */
/* -------------------------------------------------------------------------- */

__global__ void merge_basic_kernel(
    const int *A, int m,
    const int *B, int n,
    int *C
) {
    int total = m + n;
    int total_threads = gridDim.x * blockDim.x;
    int elems_per_thread = (total + total_threads - 1) / total_threads;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int k_curr = tid * elems_per_thread;
    int k_next = min((tid + 1) * elems_per_thread, total);

    if (k_curr >= total) return;

    // First co-rank call: find input range start for this thread
    int i_curr = co_rank(k_curr, A, m, B, n);
    int j_curr = k_curr - i_curr;

    // Second co-rank call: find input range end (start of next thread's range)
    int i_next = co_rank(k_next, A, m, B, n);
    int j_next = k_next - i_next;

    int len_a = i_next - i_curr;
    int len_b = j_next - j_curr;

    // Sequential merge on the thread's subarrays
    merge_sequential(&A[i_curr], len_a, &B[j_curr], len_b, &C[k_curr]);
}

/* -------------------------------------------------------------------------- */
/*  CPU Reference — sequential merge of entire arrays                         */
/* -------------------------------------------------------------------------- */

void host_merge(const int *A, int m, const int *B, int n, int *C) {
    merge_sequential(A, m, B, n, C);
}

/* -------------------------------------------------------------------------- */
/*  Helper — generate a sorted array of unique random values                  */
/* -------------------------------------------------------------------------- */

void generate_sorted_array(int *arr, int size, int max_val) {
    // Generate random values, sort, deduplicate
    for (int i = 0; i < size; i++) {
        arr[i] = rand() % max_val;
    }
    std::sort(arr, arr + size);
    // Deduplicate
    if (size > 0) {
        int *end = std::unique(arr, arr + size);
        int new_size = (int)(end - arr);
        // Fill remaining with unique values past max_val
        for (int i = new_size; i < size; i++) {
            arr[i] = max_val + i - new_size;
        }
    }
}

/* -------------------------------------------------------------------------- */
/*  Main                                                                      */
/* -------------------------------------------------------------------------- */

int main() {
    // Hardware setup: GTX 1050 is device 1
    int dev_id = 1;
    cudaSetDevice(dev_id);
    print_device_info(dev_id);

    // Problem sizes to test
    int test_sizes[] = {128, 1024, 4096, 32768, 131072, 524288};
    int num_tests = sizeof(test_sizes) / sizeof(test_sizes[0]);

    srand(time(nullptr));

    printf("\n===== Merging A + B into C =====\n");
    printf("Testing 3 configurations per size (20%% diff, 50%% diff, 80%% diff)\n\n");

    for (int t = 0; t < num_tests; t++) {
        int total = test_sizes[t];

        for (int ratio = 0; ratio < 3; ratio++) {
            // Vary the A:B size ratio
            int m, n;
            if (ratio == 0) {
                m = total / 5;               // A is small
                n = total - m;
            } else if (ratio == 1) {
                m = total / 2;               // A ≈ B
                n = total - m;
            } else {
                m = 4 * total / 5;           // A is large
                n = total - m;
            }

            int max_val = total * 4;

            // Allocate host memory
            int *h_A = new int[m];
            int *h_B = new int[n];
            int *h_C_ref = new int[total];
            int *h_C_gpu = new int[total];

            // Generate sorted inputs
            generate_sorted_array(h_A, m, max_val);
            generate_sorted_array(h_B, n, max_val);

            // CPU reference
            host_merge(h_A, m, h_B, n, h_C_ref);

            // Allocate device memory
            int *d_A, *d_B, *d_C;
            CHECK_CUDA(cudaMalloc(&d_A, m * sizeof(int)));
            CHECK_CUDA(cudaMalloc(&d_B, n * sizeof(int)));
            CHECK_CUDA(cudaMalloc(&d_C, total * sizeof(int)));

            CHECK_CUDA(cudaMemcpy(d_A, h_A, m * sizeof(int), cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMemcpy(d_B, h_B, n * sizeof(int), cudaMemcpyHostToDevice));

            // Launch basic merge kernel
            int threads = 256;
            int blocks = (total + threads * 8 - 1) / (threads * 8);  // ~8 elems/thread

            // Warm-up
            merge_basic_kernel<<<blocks, threads>>>(d_A, m, d_B, n, d_C);
            CHECK_CUDA(cudaDeviceSynchronize());

            // Timed run
            gpu_timer timer;
            timer.start();
            merge_basic_kernel<<<blocks, threads>>>(d_A, m, d_B, n, d_C);
            timer.stop();
            float ms = timer.elapsed_ms();

            CHECK_CUDA(cudaMemcpy(h_C_gpu, d_C, total * sizeof(int), cudaMemcpyDeviceToHost));

            // Validate (int comparison — exact match required for merge)
            bool pass = true;
            for (int i = 0; i < total; i++) {
                if (h_C_ref[i] != h_C_gpu[i]) {
                    printf("  FAIL: first mismatch at index %d: expected %d, got %d\n",
                           i, h_C_ref[i], h_C_gpu[i]);
                    pass = false;
                    break;
                }
            }

            float throughput = (total * sizeof(int) * 3.0f) / (ms * 1e6f);  // GB/s (read A+B, write C)
            printf("  N=%6d (m=%5d,n=%5d) | %s | %.3f ms | %.2f GB/s\n",
                   total, m, n,
                   pass ? "PASS" : "FAIL",
                   ms, throughput);

            // Cleanup
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

    printf("===== Basic Merge Kernel Complete =====\n");
    return 0;
}
