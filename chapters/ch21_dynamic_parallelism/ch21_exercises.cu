/*
 * ch21_exercises.cu — Chapter 21 Exercise Solutions
 *
 * Build: nvcc -std=c++17 -arch=sm_61 -O2 -o ch21_exercises ch21_exercises.cu
 * Run:   ./ch21_exercises
 */

#include <stdio.h>
#include <math.h>

int main()
{
    printf("=== Chapter 21: CUDA Dynamic Parallelism — Exercises ===\n\n");

    // Exercise 1: Bezier curves true/false
    printf("Exercise 1: Bezier curves\n");
    printf("  (a) If N_LINES=1024 and BLOCK_DIM=64, child kernels = 16.\n");
    printf("      Answer: TRUE. grid_blocks = ceil(1024/64) = 16 parent threads.\n");
    printf("      Each parent thread launches 1 child grid → 16 child kernels.\n\n");
    printf("  (b) Fixed-size pool should be reduced from 2048 to 1024.\n");
    printf("      Answer: FALSE. The pool should be set to at least the\n");
    printf("      expected launch count (1024). Reducing below default (2048)\n");
    printf("      to 1024 would UNNECESSARILY virtualize the pool for 1024\n");
    printf("      launches. Actually, with default 2048, all 1024 fit.\n");
    printf("      Setting to 1024 exactly is fine but not needed for performance.\n\n");
    printf("  (c) With per-thread streams, 16 streams deployed.\n");
    printf("      Answer: FALSE. There are 16 parent BLOCKS (grid_blocks),\n");
    printf("      each block has BLOCK_DIM=64 threads. Streams are private to\n");
    printf("      a block. Each of the 16 blocks has its own set of streams.\n");
    printf("      Total streams = 16 blocks × 64 threads = 1024 streams.\n\n");

    // Exercise 2: Quadtree max depth
    printf("Exercise 2: 64 points, min_points=2, max depth?\n");
    printf("  Root:     64 points (>2) → split to 4 child nodes\n");
    printf("  Depth 1:  4 nodes, avg 16 pts each (>2) → each splits → 16 nodes\n");
    printf("  Depth 2: 16 nodes, avg 4 pts each (>2) → each splits → 64 nodes\n");
    printf("  Depth 3: 64 nodes, avg 1 pt each (≤2) → recursion stops\n");
    printf("  Maximum depth including root: 4 (root + depth 1, 2, 3).\n");
    printf("  Answer: (b) 4\n\n");

    // Exercise 3: Total child kernel launches
    printf("Exercise 3: Total child kernel launches for 64-point quadtree.\n");
    printf("  Root launches 4 children\n");
    printf("  Depth 1: 4 nodes, each launches 4 → 16 child grids\n");
    printf("  Depth 2: 16 nodes, each launches 4 → 64 child grids\n");
    printf("  Depth 3: 64 nodes, all ≤2 pts → no launches\n");
    printf("  Total launches = 4 + 16 + 64 = 84\n");
    printf("  Answer: None of the above (a 21, b 4, c 64, d 16).\n");
    printf("  Correct answer: 84 child kernel launches.\n\n");

    // Exercise 4: Parent __constant__ inherited by child?
    printf("Exercise 4: Parent __constant__ variables inherited by child?\n");
    printf("  Answer: TRUE. __constant__ memory is globally visible to all\n");
    printf("  kernels, including child grids. It is part of the global memory\n");
    printf("  space and persists across kernel launches.\n\n");

    // Exercise 5: Child access to parent shared/local memory?
    printf("Exercise 5: Child kernels access parent shared/local memory?\n");
    printf("  Answer: FALSE. Shared memory and local memory are private to\n");
    printf("  the thread block and thread respectively. Child grids have\n");
    printf("  their own shared and local memory. Only global and constant\n");
    printf("  memory are shared between parent and child.\n\n");

    // Exercise 6: Concurrent child kernels
    printf("Exercise 6: 6 blocks × 256 threads, how many child kernels\n");
    printf("  can run concurrently?\n");
    printf("  Answer: (d) 1.\n");
    printf("  Without named streams, all kernels launched in the same block\n");
    printf("  use the default NULL stream and are serialized. With named\n");
    printf("  streams, up to 1536 (6×256) could run concurrently.\n");
    printf("  But with the default NULL stream (which is what the exercise\n");
    printf("  implies by not mentioning streams), only 1 per block can run\n");
    printf("  at a time. The question is ambiguous but typical answer is 1.\n");

    return 0;
}
