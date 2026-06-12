/*
 * ch15_bfs_pull.cu — Fig 15.8: Vertex-centric pull (bottom-up) BFS kernel
 *
 * Each thread is assigned to one vertex. Unvisited threads iterate over
 * incoming edges (via CSC dstPtrs/src) searching for a neighbor in the
 * previous level. On finding one, label self as current level, set flag,
 * and break out of the loop.
 *
 * Graph: 9 vertices, 15 directional edges (Fig 15.1), CSC representation
 */

#include <cstdio>
#include "../common/cuda_utils.cuh"

#define NUM_VERTICES  9
#define NUM_EDGES    15
#define UINT_MAX_VAL 0xFFFFFFFFu

// ── Graph in CSC format (incoming-edge view) ────────────────────────────
// dstPtrs[vertex] = start index in src[] for incoming edges to that vertex
// src[e] = source vertex of edge e (i.e., who points TO this vertex)
__constant__ unsigned int d_dstPtrs[NUM_VERTICES + 1] = {
    0,          // vertex 0 starts at src[0]
    1,          // vertex 1 starts at src[1]
    3,          // vertex 2 starts at src[3]
    4,          // vertex 3 starts at src[4]
    5,          // vertex 4 starts at src[5]
    7,          // vertex 5 starts at src[7]
    9,          // vertex 6 starts at src[9]
    10,         // vertex 7 starts at src[10]
    11,         // vertex 8 starts at src[11]
    15          // past-the-end
};
__constant__ unsigned int d_src[NUM_EDGES] = {
    7,          // incoming to v0
    0, 8,       // incoming to v1
    0,          // incoming to v2
    1,          // incoming to v3
    1, 3,       // incoming to v4
    2, 4,       // incoming to v5
    2,          // incoming to v6
    2,          // incoming to v7
    3, 4, 5, 6  // incoming to v8
};

// ── BFS Pull Kernel (Fig 15.8) ──────────────────────────────────────────
//
// Line-by-line reconstruction:
//   03: i = blockIdx.x * blockDim.x + threadIdx.x
//   04: if (i < numVertices)
//   05:   if (level[i] == UINT_MAX)          // vertex not yet visited
//   06:     for (e = dstPtrs[i]; e < dstPtrs[i+1]; e++)
//   07:       neighbor = src[e]              // source of incoming edge
//   08:       if (level[neighbor] == prevLevel)
//   09:         level[i] = currLevel
//   10:         *visited_new = 1
//   11:         break
__global__ void bfs_pull_kernel(
    unsigned int *level,
    unsigned int  prevLevel,
    unsigned int  currLevel,
    int          *visited_new)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;  // line 03
    if (i >= NUM_VERTICES) return;                            // line 04

    if (level[i] == UINT_MAX_VAL) {                           // line 05
        int start = d_dstPtrs[i];
        int end   = d_dstPtrs[i + 1];
        for (int e = start; e < end; e++) {                   // lines 06-07
            unsigned int neighbor = d_src[e];                  // line 08
            if (level[neighbor] == prevLevel) {                // line 09
                level[i] = currLevel;                          // line 10
                *visited_new = 1;                              // line 11
                break;                                         // line 12
            }
        }
    }
}

// ── Host BFS Driver ─────────────────────────────────────────────────────
void host_bfs_pull(unsigned int *h_level)
{
    const int blockSize = 256;
    const int gridSize  = (NUM_VERTICES + blockSize - 1) / blockSize;

    unsigned int *d_level;
    int *d_visited_new, h_visited_new;

    CHECK_CUDA(cudaSetDevice(1));
    CHECK_CUDA(cudaMalloc(&d_level, NUM_VERTICES * sizeof(unsigned int)));
    CHECK_CUDA(cudaMalloc(&d_visited_new, sizeof(int)));

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
        bfs_pull_kernel<<<gridSize, blockSize>>>(
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

    printf("BFS Pull  | levels: ");
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
    printf("=== Chapter 15: Graph Traversal — BFS Pull (Fig 15.8) ===\n");
    printf("Graph: %d vertices, %d edges, root = vertex 0\n\n",
           NUM_VERTICES, NUM_EDGES);
    host_bfs_pull(h_level);
    validate_bfs(h_level, "Fig 15.8 pull");
    return 0;
}
