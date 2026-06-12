/*
 * ch15_bfs_push.cu — Fig 15.6: Vertex-centric push (top-down) BFS kernel
 *
 * Each thread is assigned to one vertex. Threads whose vertex belongs to the
 * previous level iterate over outgoing edges (via CSR srcPtrs/dst) and label
 * all unvisited neighbors as belonging to the current level.
 *
 * Graph: 9 vertices, 15 directional edges (Fig 15.1)
 * BFS root: vertex 0
 * Expected levels: 0→lv0, {1,2}→lv1, {3,4,5,6,7}→lv2, {8}→lv3
 */

#include <cstdio>
#include "../common/cuda_utils.cuh"

#define NUM_VERTICES  9
#define NUM_EDGES    15
#define UINT_MAX_VAL 0xFFFFFFFFu

// ── Graph in CSR format (Fig 15.1) ──────────────────────────────────────
// Edges (15 total):
//   0→1, 0→2
//   1→3, 1→4
//   2→5, 2→6, 2→7
//   3→4, 3→8
//   4→5, 4→8
//   5→8
//   6→8
//   7→0
//   8→1
__constant__ unsigned int d_srcPtrs[NUM_VERTICES + 1] = {
    0, 2, 4, 7, 9, 11, 12, 13, 14, 15
};
__constant__ unsigned int d_dst[NUM_EDGES] = {
    1, 2,      // vertex 0
    3, 4,      // vertex 1
    5, 6, 7,   // vertex 2
    4, 8,      // vertex 3
    5, 8,      // vertex 4
    8,         // vertex 5
    8,         // vertex 6
    0,         // vertex 7
    1          // vertex 8
};

// ── BFS Push Kernel (Fig 15.6) ──────────────────────────────────────────
//
// Line-by-line reconstruction from textbook prose:
//   03: i = blockIdx.x * blockDim.x + threadIdx.x
//   04: if (i < numVertices)
//   05:   if (level[i] == prevLevel)
//   06:     for (int e = srcPtrs[i]; e < srcPtrs[i+1]; e++)
//   07:       unsigned int neighbor = dst[e]
//   08:       if (level[neighbor] == UINT_MAX)
//   09:         level[neighbor] = currLevel
//   10:         *visited_new = 1    // idempotent flag write
__global__ void bfs_push_kernel(
    unsigned int *level,
    unsigned int  prevLevel,
    unsigned int  currLevel,
    int          *visited_new)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;  // line 03
    if (i >= NUM_VERTICES) return;                            // line 04

    if (level[i] == prevLevel) {                              // line 05
        int start = d_srcPtrs[i];
        int end   = d_srcPtrs[i + 1];
        for (int e = start; e < end; e++) {                   // lines 06-07
            unsigned int neighbor = d_dst[e];                 // line 08
            if (level[neighbor] == UINT_MAX_VAL) {            // line 09
                level[neighbor] = currLevel;                  // line 10
                *visited_new = 1;                             // line 11 — idempotent
            }
        }
    }
}

// ── Host BFS Driver ─────────────────────────────────────────────────────
void host_bfs_push(unsigned int *h_level)
{
    const int blockSize = 256;
    const int gridSize  = (NUM_VERTICES + blockSize - 1) / blockSize;

    unsigned int *d_level;
    int *d_visited_new, h_visited_new;

    CHECK_CUDA(cudaSetDevice(1));
    CHECK_CUDA(cudaMalloc(&d_level, NUM_VERTICES * sizeof(unsigned int)));
    CHECK_CUDA(cudaMalloc(&d_visited_new, sizeof(int)));

    // Initialize: all vertices unvisited, root (vertex 0) at level 0
    unsigned int h_level_init[NUM_VERTICES];
    for (int i = 0; i < NUM_VERTICES; i++)
        h_level_init[i] = UINT_MAX_VAL;
    h_level_init[0] = 0;  // root

    CHECK_CUDA(cudaMemcpy(d_level, h_level_init,
                          NUM_VERTICES * sizeof(unsigned int),
                          cudaMemcpyHostToDevice));

    unsigned int currLevel = 1;
    gpu_timer timer;
    timer.start();

    while (true) {
        CHECK_CUDA(cudaMemset(d_visited_new, 0, sizeof(int)));
        bfs_push_kernel<<<gridSize, blockSize>>>(
            d_level, currLevel - 1, currLevel, d_visited_new);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaMemcpy(&h_visited_new, d_visited_new, sizeof(int),
                              cudaMemcpyDeviceToHost));
        if (h_visited_new == 0) break;
        currLevel++;
    }

    timer.stop();

    CHECK_CUDA(cudaMemcpy(h_level, d_level,
                          NUM_VERTICES * sizeof(unsigned int),
                          cudaMemcpyDeviceToHost));

    printf("BFS Push  | levels: ");
    for (int i = 0; i < NUM_VERTICES; i++)
        printf("%u ", h_level[i]);
    printf("| time: %.3f ms\n", timer.elapsed_ms());

    CHECK_CUDA(cudaFree(d_level));
    CHECK_CUDA(cudaFree(d_visited_new));
}

// ── Validation ──────────────────────────────────────────────────────────
void validate_bfs(unsigned int *h_level, const char *kernel_name)
{
    unsigned int expected[NUM_VERTICES] = {
        0,          // vertex 0: root, level 0
        1,          // vertex 1: level 1
        1,          // vertex 2: level 1
        2,          // vertex 3: level 2
        2,          // vertex 4: level 2
        2,          // vertex 5: level 2
        2,          // vertex 6: level 2
        2,          // vertex 7: level 2
        3           // vertex 8: level 3
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

// ── Main ────────────────────────────────────────────────────────────────
int main()
{
    unsigned int h_level[NUM_VERTICES];

    printf("=== Chapter 15: Graph Traversal — BFS Push (Fig 15.6) ===\n");
    printf("Graph: %d vertices, %d edges, root = vertex 0\n\n",
           NUM_VERTICES, NUM_EDGES);

    host_bfs_push(h_level);
    validate_bfs(h_level, "Fig 15.6 push");

    return 0;
}
