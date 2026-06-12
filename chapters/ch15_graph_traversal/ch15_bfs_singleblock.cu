/*
 * ch15_bfs_singleblock.cu — Exercise 3: Single-block multi-level BFS kernel
 *
 * When frontiers are small, grid launch overhead dominates. This kernel
 * processes multiple consecutive BFS levels in a single block using only
 * a shared-memory local frontier and __syncthreads() between levels.
 *
 * When the frontier grows beyond the shared-memory capacity, the kernel
 * copies the remaining frontier to global memory and returns control to
 * the host, which continues with the regular multi-grid kernel.
 *
 * CSR format for outgoing-edge access.
 */

#include <cstdio>
#include "../common/cuda_utils.cuh"

#define NUM_VERTICES         9
#define NUM_EDGES           15
#define UINT_MAX_VAL        0xFFFFFFFFu
#define BLOCK_FRONTIER_CAP  64   // max frontier entries in shared memory

// ── CSR Graph Data ─────────────────────────────────────────────────────
__constant__ unsigned int d_srcPtrs[NUM_VERTICES + 1] = {
    0, 2, 4, 7, 9, 11, 12, 13, 14, 15
};
__constant__ unsigned int d_dst[NUM_EDGES] = {
    1, 2, 3, 4, 5, 6, 7, 4, 8, 5, 8, 8, 8, 0, 1
};

// ── Single-Block Multi-Level BFS Kernel (Exercise 3) ────────────────────
//
// Processes multiple levels in one block launch:
//   - Maintains local frontier in shared memory
//   - Between levels, __syncthreads() for synchronization
//   - If frontier outgrows capacity, flushes to global and exits
//   - Returns level reached + frontier size to host
//
__global__ void bfs_singleblock_kernel(
    unsigned int *level,
    unsigned int *globalFrontier,
    unsigned int *numGlobalFrontier,
    unsigned int *currentLevel,       // in/out: starting level, updated on exit
    unsigned int  startLevel)
{
    __shared__ unsigned int s_frontier[BLOCK_FRONTIER_CAP];
    __shared__ unsigned int s_count;
    __shared__ unsigned int s_nextCount;
    __shared__ unsigned int s_overflown;   // set to 1 if capacity exceeded

    unsigned int currLevel = startLevel;

    // Thread 0 copies the initial frontier from global memory
    // (already set up by host — root vertex at start)
    if (threadIdx.x == 0) {
        s_count = *numGlobalFrontier;
        s_overflown = 0;
    }
    __syncthreads();

    // Copy initial frontier from global to shared
    for (int j = threadIdx.x; j < s_count; j += blockDim.x) {
        s_frontier[j] = globalFrontier[j];
    }
    __syncthreads();

    while (true) {
        // ── Process current level: label neighbors ────────────────────
        s_nextCount = 0;
        __syncthreads();

        for (int k = threadIdx.x; k < s_count; k += blockDim.x) {
            unsigned int v = s_frontier[k];
            int start = d_srcPtrs[v];
            int end   = d_srcPtrs[v + 1];
            for (int e = start; e < end; e++) {
                unsigned int nbr = d_dst[e];
                unsigned int old = atomicCAS(&level[nbr],
                                           UINT_MAX_VAL, currLevel);
                if (old == UINT_MAX_VAL) {
                    unsigned int idx = atomicAdd(&s_nextCount, 1u);
                    if (idx < BLOCK_FRONTIER_CAP - 1) {
                        // Note: we use a separate staging area or write
                        // to a second buffer to avoid overwriting s_frontier
                        // while other threads may still be reading it.
                        // Write to global directly for simplicity since
                        // it's a small graph. A proper implementation would
                        // use double-buffering in shared memory.
                        globalFrontier[idx] = nbr;
                    } else {
                        s_overflown = 1;
                        atomicSub(&s_nextCount, 1u);
                    }
                }
            }
        }
        __syncthreads();

        if (s_overflown) break;

        if (s_nextCount == 0) break;  // BFS complete

        // ── Advance to next level: rotate s_frontier ─────────────────
        // Copy new frontier from global to shared
        for (int j = threadIdx.x; j < s_nextCount; j += blockDim.x) {
            s_frontier[j] = globalFrontier[j];
        }
        s_count = s_nextCount;
        currLevel++;
        __syncthreads();
    }

    // ── Write results back to global ──────────────────────────────────
    if (threadIdx.x == 0) {
        *numGlobalFrontier = s_overflown ? s_count : s_nextCount;
        *currentLevel = currLevel;
    }
}

// ── Regular multi-grid frontier-based kernel (continuation) ─────────────
// (simplified — uses full-vertex-scan push from Fig 15.6 for remaining levels)
__global__ void bfs_push_kernel(
    unsigned int *level,
    unsigned int  prevLevel,
    unsigned int  currLevel,
    int          *visited_new)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= NUM_VERTICES) return;
    if (level[i] == prevLevel) {
        int start = d_srcPtrs[i];
        int end   = d_srcPtrs[i + 1];
        for (int e = start; e < end; e++) {
            unsigned int nbr = d_dst[e];
            if (level[nbr] == UINT_MAX_VAL) {
                level[nbr] = currLevel;
                *visited_new = 1;
            }
        }
    }
}

// ── Host BFS with Single-Block Multi-Level Kernel ────────────────────────
void host_bfs_singleblock(unsigned int *h_level)
{
    const int blockSize = 64;

    unsigned int *d_level, *d_globalFrontier, *d_numGFrontier, *d_currentLevel;
    int *d_visited_new;
    CHECK_CUDA(cudaSetDevice(1));
    CHECK_CUDA(cudaMalloc(&d_level, NUM_VERTICES * sizeof(unsigned int)));
    CHECK_CUDA(cudaMalloc(&d_globalFrontier, NUM_VERTICES * sizeof(unsigned int)));
    CHECK_CUDA(cudaMalloc(&d_numGFrontier, sizeof(unsigned int)));
    CHECK_CUDA(cudaMalloc(&d_currentLevel, sizeof(unsigned int)));
    CHECK_CUDA(cudaMalloc(&d_visited_new, sizeof(int)));

    // Initialize levels
    unsigned int h_level_init[NUM_VERTICES];
    for (int i = 0; i < NUM_VERTICES; i++)
        h_level_init[i] = UINT_MAX_VAL;
    h_level_init[0] = 0;
    CHECK_CUDA(cudaMemcpy(d_level, h_level_init,
                          NUM_VERTICES * sizeof(unsigned int),
                          cudaMemcpyHostToDevice));

    // Initial frontier: just root (vertex 0)
    unsigned int h_frontier_init[1] = {0};
    unsigned int h_numInit = 1;
    CHECK_CUDA(cudaMemcpy(d_globalFrontier, h_frontier_init,
                          sizeof(unsigned int),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_numGFrontier, &h_numInit, sizeof(unsigned int),
                          cudaMemcpyHostToDevice));

    // Starting level for the single-block kernel
    unsigned int startLevel = 1;
    unsigned int h_startLevel = startLevel;
    CHECK_CUDA(cudaMemcpy(d_currentLevel, &h_startLevel, sizeof(unsigned int),
                          cudaMemcpyHostToDevice));

    gpu_timer timer;
    timer.start();

    // Phase 1: Single-block kernel for small-frontier levels
    bfs_singleblock_kernel<<<1, blockSize>>>(
        d_level, d_globalFrontier, d_numGFrontier,
        d_currentLevel, startLevel);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    unsigned int h_currLevel, h_numFrontier;
    CHECK_CUDA(cudaMemcpy(&h_currLevel, d_currentLevel, sizeof(unsigned int),
                          cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(&h_numFrontier, d_numGFrontier, sizeof(unsigned int),
                          cudaMemcpyDeviceToHost));

    // Phase 2: Regular multi-grid kernel for remaining levels
    unsigned int currLevel = h_currLevel;
    if (h_numFrontier > 0) {
        // Continue with full-grid push for remaining levels
        // First, if the single-block left us mid-level, we need to fixup:
        // The single-block may have written new neighbors to level[] but
        // not set *visited_new for the host loop. We need to check if BFS
        // is truly done. The single-block stopped either because:
        //   a) s_nextCount == 0 → BFS complete, no more levels
        //   b) s_overflown → frontier overflowed, continue with regular kernel
        //
        // For case (b): currLevel already points to the NEXT level to process.
        // The frontier was not fully processed — we need to label the 
        // overflow frontier's neighbors at currLevel, then continue.

        // Actually, when overflown: s_count is the level that overflowed,
        // and we haven't processed its neighbors yet. So we continue from
        // currLevel (which is the same level — the kernel didn't advance).

        if (h_numFrontier > 0 && currLevel > 0) {
            // Fall back to full-grid push for remaining levels
            int gridSize = (NUM_VERTICES + blockSize - 1) / blockSize;
            int h_visited_new;

            while (true) {
                CHECK_CUDA(cudaMemset(d_visited_new, 0, sizeof(int)));
                bfs_push_kernel<<<gridSize, blockSize>>>(
                    d_level, currLevel - 1, currLevel, d_visited_new);
                CHECK_CUDA(cudaGetLastError());
                CHECK_CUDA(cudaDeviceSynchronize());

                CHECK_CUDA(cudaMemcpy(&h_visited_new, d_visited_new,
                                      sizeof(int), cudaMemcpyDeviceToHost));
                if (h_visited_new == 0) break;
                currLevel++;
            }
        }
    }

    timer.stop();

    CHECK_CUDA(cudaMemcpy(h_level, d_level,
                          NUM_VERTICES * sizeof(unsigned int),
                          cudaMemcpyDeviceToHost));

    printf("BFS 1-Block | levels: ");
    for (int i = 0; i < NUM_VERTICES; i++)
        printf("%u ", h_level[i]);
    printf("| time: %.3f ms\n", timer.elapsed_ms());

    CHECK_CUDA(cudaFree(d_level));
    CHECK_CUDA(cudaFree(d_globalFrontier));
    CHECK_CUDA(cudaFree(d_numGFrontier));
    CHECK_CUDA(cudaFree(d_currentLevel));
    CHECK_CUDA(cudaFree(d_visited_new));
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
    printf("=== Chapter 15: Single-Block Multi-Level BFS (Exercise 3) ===\n");
    printf("Graph: %d vertices, %d edges, root = vertex 0\n\n",
           NUM_VERTICES, NUM_EDGES);
    host_bfs_singleblock(h_level);
    validate_bfs(h_level, "Ex3 single-block");
    return 0;
}