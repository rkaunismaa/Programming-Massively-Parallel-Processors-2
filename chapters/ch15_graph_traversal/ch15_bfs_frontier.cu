/*
 * ch15_bfs_frontier.cu — Fig 15.12: Vertex-centric push BFS with frontiers
 *
 * Instead of launching one thread per vertex (most of which do nothing),
 * only vertices in the previous frontier are processed. Uses atomicCAS
 * for visit-check-and-label to prevent redundant frontier insertion.
 *
 * CSR format for outgoing-edge access.
 */

#include <cstdio>
#include "../common/cuda_utils.cuh"

#define NUM_VERTICES   9
#define NUM_EDGES     15
#define UINT_MAX_VAL  0xFFFFFFFFu

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

// ── BFS Push with Frontiers Kernel (Fig 15.12) ──────────────────────────
//
// Reconstructed from text:
//   05: i = blockIdx.x * blockDim.x + threadIdx.x
//   06: if (i < numPrevFrontier)
//   07:   v = prevFrontier[i]              // vertex to process
//   08:   for (e = srcPtrs[v]; e < srcPtrs[v+1]; e++)
//   09:     neighbor = dst[e]
//   10:     old = atomicCAS(&level[neighbor], UINT_MAX, currLevel)
//   11:     if (old == UINT_MAX)              // CAS succeeded — first visit
//   12:       idx = atomicAdd(&numCurrFrontier, 1)
//   13:       currFrontier[idx] = neighbor
__global__ void bfs_frontier_kernel(
    unsigned int *level,
    unsigned int *prevFrontier,
    unsigned int *currFrontier,
    unsigned int  numPrevFrontier,
    unsigned int *numCurrFrontier,
    unsigned int  currLevel)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;  // line 05
    if (i >= numPrevFrontier) return;                          // line 06

    unsigned int v = prevFrontier[i];                        // line 07

    int start = d_srcPtrs[v];
    int end   = d_srcPtrs[v + 1];
    for (int e = start; e < end; e++) {                       // lines 08-09
        unsigned int neighbor = d_dst[e];                     // line 10

        unsigned int old = atomicCAS(&level[neighbor],          // line 11
                                   UINT_MAX_VAL, currLevel);
        if (old == UINT_MAX_VAL) {                           // CAS succeeded
            unsigned int idx = atomicAdd(numCurrFrontier, 1u);  // line 12
            currFrontier[idx] = neighbor;                     // line 13
        }
    }
}

// ── Host BFS with Frontiers ──────────────────────────────────────────────
void host_bfs_frontier(unsigned int *h_level)
{
    const int blockSize = 256;

    // Device allocations
    unsigned int *d_level, *d_prevFrontier, *d_currFrontier, *d_numCurrFrontier;
    CHECK_CUDA(cudaSetDevice(1));
    CHECK_CUDA(cudaMalloc(&d_level, NUM_VERTICES * sizeof(unsigned int)));
    CHECK_CUDA(cudaMalloc(&d_prevFrontier, NUM_VERTICES * sizeof(unsigned int)));
    CHECK_CUDA(cudaMalloc(&d_currFrontier, NUM_VERTICES * sizeof(unsigned int)));
    CHECK_CUDA(cudaMalloc(&d_numCurrFrontier, sizeof(unsigned int)));

    // Initialize levels
    unsigned int h_level_init[NUM_VERTICES];
    unsigned int h_prevFrontier[NUM_VERTICES];
    for (int i = 0; i < NUM_VERTICES; i++)
        h_level_init[i] = UINT_MAX_VAL;
    h_level_init[0] = 0;

    // Initial frontier: just the root (vertex 0)
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
        // Reset current frontier counter
        unsigned int h_zero = 0;
        CHECK_CUDA(cudaMemcpy(d_numCurrFrontier, &h_zero, sizeof(unsigned int),
                              cudaMemcpyHostToDevice));

        int gridSize = (numPrev + blockSize - 1) / blockSize;
        bfs_frontier_kernel<<<gridSize, blockSize>>>(
            d_level, d_prevFrontier, d_currFrontier,
            numPrev, d_numCurrFrontier, currLevel);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());

        unsigned int h_numCurr;
        CHECK_CUDA(cudaMemcpy(&h_numCurr, d_numCurrFrontier, sizeof(unsigned int),
                              cudaMemcpyDeviceToHost));

        if (h_numCurr == 0) break;

        // currFrontier becomes prevFrontier for next iteration
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

    printf("BFS Frontier | levels: ");
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
    printf("=== Chapter 15: BFS with Frontiers (Fig 15.12) ===\n");
    printf("Graph: %d vertices, %d edges, root = vertex 0\n\n",
           NUM_VERTICES, NUM_EDGES);
    host_bfs_frontier(h_level);
    validate_bfs(h_level, "Fig 15.12 frontier");
    return 0;
}
