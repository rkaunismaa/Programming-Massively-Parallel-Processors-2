/*
 * ch15_bfs_direction_optimized.cu — Exercise 2: Direction-optimized BFS
 *
 * Switches between push (top-down, CSR) for early levels and pull
 * (bottom-up, CSC) for later levels. Push uses the vertex-centric
 * kernel from Fig 15.6; pull uses the kernel from Fig 15.8.
 *
 * Switch heuristic: numFrontier * alpha > numUnvisited → switch to pull.
 * Lower alpha (e.g., 1.0) switches earlier, useful for high-degree graphs.
 */

#include <cstdio>
#include "../common/cuda_utils.cuh"

#define NUM_VERTICES   9
#define NUM_EDGES     15
#define UINT_MAX_VAL  0xFFFFFFFFu
#define SWITCH_ALPHA  1.0f

// ── CSR Graph Data (push / top-down) ────────────────────────────────────
__constant__ unsigned int d_srcPtrs[NUM_VERTICES + 1] = {
    0, 2, 4, 7, 9, 11, 12, 13, 14, 15
};
__constant__ unsigned int d_dst[NUM_EDGES] = {
    1, 2, 3, 4, 5, 6, 7, 4, 8, 5, 8, 8, 8, 0, 1
};

// ── CSC Graph Data (pull / bottom-up) ───────────────────────────────────
__constant__ unsigned int d_dstPtrs[NUM_VERTICES + 1] = {
    0, 1, 3, 4, 5, 7, 9, 10, 11, 15
};
__constant__ unsigned int d_src[NUM_EDGES] = {
    7, 0, 8, 0, 1, 1, 3, 2, 4, 2, 2, 3, 4, 5, 6
};

// ── Push kernel (Fig 15.6) ──────────────────────────────────────────────
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

// ── Pull kernel (Fig 15.8) ──────────────────────────────────────────────
__global__ void bfs_pull_kernel(
    unsigned int *level,
    unsigned int  prevLevel,
    unsigned int  currLevel,
    int          *visited_new)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= NUM_VERTICES) return;
    if (level[i] == UINT_MAX_VAL) {
        int start = d_dstPtrs[i];
        int end   = d_dstPtrs[i + 1];
        for (int e = start; e < end; e++) {
            unsigned int nbr = d_src[e];
            if (level[nbr] == prevLevel) {
                level[i] = currLevel;
                *visited_new = 1;
                break;
            }
        }
    }
}

// ── Host Direction-Optimized BFS ─────────────────────────────────────────
void host_bfs_direction_opt(unsigned int *h_level)
{
    const int blockSize = 256;

    unsigned int *d_level;
    int *d_visited_new;
    CHECK_CUDA(cudaSetDevice(1));
    CHECK_CUDA(cudaMalloc(&d_level, NUM_VERTICES * sizeof(unsigned int)));
    CHECK_CUDA(cudaMalloc(&d_visited_new, sizeof(int)));

    // Initialize: root (0) at level 0, rest unvisited
    unsigned int h_level_init[NUM_VERTICES];
    for (int i = 0; i < NUM_VERTICES; i++)
        h_level_init[i] = UINT_MAX_VAL;
    h_level_init[0] = 0;
    CHECK_CUDA(cudaMemcpy(d_level, h_level_init,
                          NUM_VERTICES * sizeof(unsigned int),
                          cudaMemcpyHostToDevice));

    // Count visited (just root initially) and frontier size
    unsigned int numVisited = 1;
    unsigned int prevFrontierSize = 1;  // just root at level 0
    bool usePush = true;

    unsigned int currLevel = 1;
    int h_visited_new;
    gpu_timer timer;
    timer.start();

    while (true) {
        // Decide direction for THIS level
        unsigned int numUnvisited = NUM_VERTICES - numVisited;
        if (usePush && (float)prevFrontierSize * SWITCH_ALPHA > (float)numUnvisited) {
            usePush = false;
            printf("  [switching to pull at level %u: frontier=%u, unvisited=%u]\n",
                   currLevel - 1, prevFrontierSize, numUnvisited);
        }

        CHECK_CUDA(cudaMemset(d_visited_new, 0, sizeof(int)));

        if (usePush) {
            int gridSize = (NUM_VERTICES + blockSize - 1) / blockSize;
            bfs_push_kernel<<<gridSize, blockSize>>>(
                d_level, currLevel - 1, currLevel, d_visited_new);
        } else {
            int gridSize = (NUM_VERTICES + blockSize - 1) / blockSize;
            bfs_pull_kernel<<<gridSize, blockSize>>>(
                d_level, currLevel - 1, currLevel, d_visited_new);
        }

        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaMemcpy(&h_visited_new, d_visited_new, sizeof(int),
                              cudaMemcpyDeviceToHost));
        if (h_visited_new == 0) break;

        // Count new frontier size for next iteration's heuristic
        CHECK_CUDA(cudaMemcpy(h_level, d_level,
                              NUM_VERTICES * sizeof(unsigned int),
                              cudaMemcpyDeviceToHost));
        prevFrontierSize = 0;
        for (int i = 0; i < NUM_VERTICES; i++)
            if (h_level[i] == currLevel)
                prevFrontierSize++;
        numVisited += prevFrontierSize;

        currLevel++;
    }

    timer.stop();

    CHECK_CUDA(cudaMemcpy(h_level, d_level,
                          NUM_VERTICES * sizeof(unsigned int),
                          cudaMemcpyDeviceToHost));

    printf("BFS DirOpt | levels: ");
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
    printf("=== Chapter 15: Direction-Optimized BFS (Exercise 2) ===\n");
    printf("Graph: %d vertices, %d edges, root = vertex 0\n\n",
           NUM_VERTICES, NUM_EDGES);
    host_bfs_direction_opt(h_level);
    validate_bfs(h_level, "Ex2 direction-opt");
    return 0;
}
