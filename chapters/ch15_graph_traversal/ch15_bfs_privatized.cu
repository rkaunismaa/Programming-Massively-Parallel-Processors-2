/*
 * ch15_bfs_privatized.cu — Fig 15.14: Vertex-centric push BFS
 * with privatized frontiers (shared-memory per-block frontier)
 *
 * Each block maintains a private local frontier in shared memory.
 * When the local frontier fills, overflow goes to the global frontier.
 * After all threads finish, the local frontier is coalesced-written
 * to global memory via a single atomicAdd allocation.
 *
 * CSR format for outgoing-edge access.
 */

#include <cstdio>
#include "../common/cuda_utils.cuh"

#define NUM_VERTICES        9
#define NUM_EDGES          15
#define UINT_MAX_VAL       0xFFFFFFFFu
#define BLOCK_FRONTIER_CAP 64   // max frontier entries per block in shared memory

// ── CSR Graph Data ─────────────────────────────────────────────────────
__constant__ unsigned int d_srcPtrs[NUM_VERTICES + 1] = {
    0, 2, 4, 7, 9, 11, 12, 13, 14, 15
};
__constant__ unsigned int d_dst[NUM_EDGES] = {
    1, 2,
    3, 4,
    5, 6, 7,
    4, 8,
    5, 8,
    8,
    8,
    0,
    1
};

// ── BFS Push with Privatized Frontiers Kernel (Fig 15.14) ──────────────
//
// Reconstructed from pages 359-361 text:
//   07-08: __shared__ unsigned int s_frontier[CAP], s_count
//   09-11: if (tid==0) s_count=0; __syncthreads()
//   12-16: thread assignment, bounds check, load vertex
//   17:    v = prevFrontier[i]
//   18-19: for (e = srcPtrs[v]; e < srcPtrs[v+1]; e++)
//   20:      neighbor = dst[e]
//   21:      old = atomicCAS(&level[neighbor], UINT_MAX, currLevel)
//   22:      if (old == UINT_MAX)
//   23:        idx = atomicAdd(&s_count, 1)
//   24:        if (idx < CAP)
//   25:          s_frontier[idx] = neighbor
//   26:        else
//   27:          atomicSub(&s_count, 1)           // rollback
//   28:          gidx = atomicAdd(numCurr, 1)     // global fallback
//   29:          currFrontier[gidx] = neighbor
//   33: __syncthreads()  — wait for all in block
//   36-39: if (tid==0) s_offset = atomicAdd(numCurr, s_count); __syncthreads()
//   43-46: for (j=tid; j<s_count; j+=blockDim) currFrontier[s_offset+j] = s_frontier[j]
//
__global__ void bfs_privatized_kernel(
    unsigned int *level,
    unsigned int *prevFrontier,
    unsigned int *currFrontier,
    unsigned int  numPrevFrontier,
    unsigned int *numCurrFrontier,
    unsigned int  currLevel)
{
    // ── Per-block private frontier in shared memory ───────────────────
    __shared__ unsigned int s_frontier[BLOCK_FRONTIER_CAP];  // line 07
    __shared__ unsigned int s_count;                         // line 08

    // One thread initializes the counter
    if (threadIdx.x == 0)                                    // line 09
        s_count = 0;                                         // line 10
    __syncthreads();                                         // line 11

    // ── Thread assignment and edge processing ─────────────────────────
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;  // line 12
    if (i < numPrevFrontier) {                               // line 13
        unsigned int v = prevFrontier[i];                    // line 17

        int start = d_srcPtrs[v];
        int end   = d_srcPtrs[v + 1];
        for (int e = start; e < end; e++) {                  // lines 18-19
            unsigned int neighbor = d_dst[e];                // line 20

            unsigned int old = atomicCAS(&level[neighbor],    // line 21
                                       UINT_MAX_VAL, currLevel);
            if (old == UINT_MAX_VAL) {                       // line 22
                unsigned int idx = atomicAdd(&s_count, 1u);   // line 23
                if (idx < BLOCK_FRONTIER_CAP) {              // line 24
                    s_frontier[idx] = neighbor;              // line 25
                } else {
                    atomicSub(&s_count, 1u);                 // line 27 — rollback
                    unsigned int gidx = atomicAdd(numCurrFrontier, 1u);  // line 28
                    currFrontier[gidx] = neighbor;           // line 29
                }
            }
        }
    }

    __syncthreads();                                         // line 33

    // ── Flush private frontier to global ──────────────────────────────
    unsigned int s_offset = 0;
    if (threadIdx.x == 0) {                                  // line 36
        if (s_count > 0) {
            s_offset = atomicAdd(numCurrFrontier, s_count);  // line 38
        }
    }
    __syncthreads();                                         // line 40

    // Coalesced copy from shared to global frontier
    for (unsigned int j = threadIdx.x; j < s_count; j += blockDim.x) {  // lines 43-44
        currFrontier[s_offset + j] = s_frontier[j];         // lines 45-46
    }
}

// ── Host BFS with Privatized Frontiers ───────────────────────────────────
void host_bfs_privatized(unsigned int *h_level)
{
    const int blockSize = 64;  // small to test privatization

    unsigned int *d_level, *d_prevFrontier, *d_currFrontier, *d_numCurrFrontier;
    CHECK_CUDA(cudaSetDevice(1));
    CHECK_CUDA(cudaMalloc(&d_level, NUM_VERTICES * sizeof(unsigned int)));
    CHECK_CUDA(cudaMalloc(&d_prevFrontier, NUM_VERTICES * sizeof(unsigned int)));
    CHECK_CUDA(cudaMalloc(&d_currFrontier, NUM_VERTICES * sizeof(unsigned int)));
    CHECK_CUDA(cudaMalloc(&d_numCurrFrontier, sizeof(unsigned int)));

    // Initialize levels
    unsigned int h_level_init[NUM_VERTICES];
    for (int i = 0; i < NUM_VERTICES; i++)
        h_level_init[i] = UINT_MAX_VAL;
    h_level_init[0] = 0;

    unsigned int h_prevFrontier[NUM_VERTICES];
    h_prevFrontier[0] = 0;
    unsigned int numPrev = 1;

    CHECK_CUDA(cudaMemcpy(d_level, h_level_init,
                          NUM_VERTICES * sizeof(unsigned int),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_prevFrontier, h_prevFrontier,
                          numPrev * sizeof(unsigned int),
                          cudaMemcpyHostToDevice));

    unsigned int currLevel = 1;
    gpu_timer timer;
    timer.start();

    while (true) {
        unsigned int h_zero = 0;
        CHECK_CUDA(cudaMemcpy(d_numCurrFrontier, &h_zero, sizeof(unsigned int),
                              cudaMemcpyHostToDevice));

        int gridSize = (numPrev + blockSize - 1) / blockSize;
        bfs_privatized_kernel<<<gridSize, blockSize>>>(
            d_level, d_prevFrontier, d_currFrontier,
            numPrev, d_numCurrFrontier, currLevel);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());

        unsigned int h_numCurr;
        CHECK_CUDA(cudaMemcpy(&h_numCurr, d_numCurrFrontier, sizeof(unsigned int),
                              cudaMemcpyDeviceToHost));

        if (h_numCurr == 0) break;

        numPrev = h_numCurr;
        CHECK_CUDA(cudaMemcpy(d_prevFrontier, d_currFrontier,
                              numPrev * sizeof(unsigned int),
                              cudaMemcpyDeviceToDevice));
        currLevel++;
    }

    timer.stop();

    CHECK_CUDA(cudaMemcpy(h_level, d_level,
                          NUM_VERTICES * sizeof(unsigned int),
                          cudaMemcpyDeviceToHost));

    printf("BFS Privat  | levels: ");
    for (int i = 0; i < NUM_VERTICES; i++)
        printf("%u ", h_level[i]);
    printf("| time: %.3f ms\n", timer.elapsed_ms());

    CHECK_CUDA(cudaFree(d_level));
    CHECK_CUDA(cudaFree(d_prevFrontier));
    CHECK_CUDA(cudaFree(d_currFrontier));
    CHECK_CUDA(cudaFree(d_numCurrFrontier));
}

void validate_bfs(unsigned int *h_level, const char *kernel_name)
{
    unsigned int expected[NUM_VERTICES] = {
        0, 1, 1, 2, 2, 2, 2, 2, 3
    };
    bool pass = true;
    for (int i = 0; i < NUM_VERTICES; i++) {
        if (h_level[i] != expected[i]) {
            printf("  FAIL at vertex %d: expected level %u, got %u\n",
                   i, expected[i], h_level[i]);
            pass = false;
        }
    }
    printf("Validation [%s]: %s\n", kernel_name, pass ? "PASS" : "FAIL");
}

int main()
{
    unsigned int h_level[NUM_VERTICES];
    printf("=== Chapter 15: BFS with Privatized Frontiers (Fig 15.14) ===\n");
    printf("Graph: %d vertices, %d edges, root = vertex 0\n\n",
           NUM_VERTICES, NUM_EDGES);
    host_bfs_privatized(h_level);
    validate_bfs(h_level, "Fig 15.14 privatized");
    return 0;
}
