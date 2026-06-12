/*
 * ch15_bfs_edge.cu — Fig 15.10: Edge-centric BFS kernel
 *
 * Each thread is assigned to one edge. Thread checks whether the source
 * vertex belongs to the previous level and the destination is unvisited.
 * If so, labels the destination as current level.
 *
 * Uses COO representation of the graph.
 */

#include <cstdio>
#include "../common/cuda_utils.cuh"

#define NUM_VERTICES  9
#define NUM_EDGES    15
#define UINT_MAX_VAL 0xFFFFFFFFu

// ── Graph in COO format ─────────────────────────────────────────────────
__constant__ unsigned int d_src[NUM_EDGES] = {
    0, 0,                     // edges 0-1
    1, 1,                     // edges 2-3
    2, 2, 2,                  // edges 4-6
    3, 3,                     // edges 7-8
    4, 4,                     // edges 9-10
    5,                        // edge 11
    6,                        // edge 12
    7,                        // edge 13
    8                         // edge 14
};
__constant__ unsigned int d_dst[NUM_EDGES] = {
    1, 2,                     // 0→1, 0→2
    3, 4,                     // 1→3, 1→4
    5, 6, 7,                  // 2→5, 2→6, 2→7
    4, 8,                     // 3→4, 3→8
    5, 8,                     // 4→5, 4→8
    8,                        // 5→8
    8,                        // 6→8
    0,                        // 7→0
    1                         // 8→1
};

// ── BFS Edge-Centric Kernel (Fig 15.10) ─────────────────────────────────
//
// Line-by-line reconstruction:
//   03: i = blockIdx.x * blockDim.x + threadIdx.x
//   04: if (i < numEdges)
//   05:   src_vertex = src[i]
//   06:   if (level[src_vertex] == prevLevel)
//   07:     neighbor = dst[i]
//   08:     if (level[neighbor] == UINT_MAX)
//   09:       level[neighbor] = currLevel
//   10:       *visited_new = 1
__global__ void bfs_edge_kernel(
    unsigned int *level,
    unsigned int  prevLevel,
    unsigned int  currLevel,
    int          *visited_new)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;          // line 03
    if (i >= NUM_EDGES) return;                             // line 04

    unsigned int src_vertex = d_src[i];                     // line 05
    if (level[src_vertex] == prevLevel) {                   // line 06
        unsigned int neighbor = d_dst[i];                   // line 07
        if (level[neighbor] == UINT_MAX_VAL) {             // line 08
            level[neighbor] = currLevel;                    // line 09
            *visited_new = 1;                               // line 10
        }
    }
}

// ── Host BFS Driver ─────────────────────────────────────────────────────
void host_bfs_edge(unsigned int *h_level)
{
    const int blockSize = 256;
    const int gridSize  = (NUM_EDGES + blockSize - 1) / blockSize;

    unsigned int *d_level;
    int *d_visited_new, h_visited_new;

    CHECK_CUDA(cudaSetDevice(1));
    CHECK_CUDA(cudaMalloc(&d_level, NUM_VERTICES * sizeof(unsigned int)));
    CHECK_CUDA(cudaMalloc(&d_visited_new, sizeof(int)));

    unsigned int h_level_init[NUM_VERTICES];
    for (int i = 0; i < NUM_VERTICES; i++)
        h_level_init[i] = UINT_MAX_VAL;
    h_level_init[0] = 0;

    CHECK_CUDA(cudaMemcpy(d_level, h_level_init,
                          NUM_VERTICES * sizeof(unsigned int),
                          cudaMemcpyHostToDevice));

    unsigned int currLevel = 1;
    gpu_timer timer;
    timer.start();

    while (true) {
        CHECK_CUDA(cudaMemset(d_visited_new, 0, sizeof(int)));
        bfs_edge_kernel<<<gridSize, blockSize>>>(
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

    printf("BFS Edge  | levels: ");
    for (int i = 0; i < NUM_VERTICES; i++)
        printf("%u ", h_level[i]);
    printf("| time: %.3f ms\n", timer.elapsed_ms());

    CHECK_CUDA(cudaFree(d_level));
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
    printf("=== Chapter 15: Graph Traversal — BFS Edge-Centric (Fig 15.10) ===\n");
    printf("Graph: %d vertices, %d edges, root = vertex 0\n\n",
           NUM_VERTICES, NUM_EDGES);
    host_bfs_edge(h_level);
    validate_bfs(h_level, "Fig 15.10 edge-centric");
    return 0;
}
