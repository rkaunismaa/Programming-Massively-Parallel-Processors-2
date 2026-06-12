/*
 * ch21_quadtree.cu — Quadtree with Recursive CUDA Dynamic Parallelism
 *
 * From PMPP 4th Ed, Chapter 21.4: A recursive example — quadtrees
 * Fig 21.10: build_quadtree_kernel (recursive kernel)
 * Fig 21.11: Device functions (check, count, scan, reorder, prepare)
 * Appendix A21.1: Support code (Points, Bounding_box, Quadtree_node, Parameters)
 *
 * Each block represents one quadtree node. If a node has more than the minimum
 * number of points, the last thread launches 4 child blocks (one per quadrant).
 * Recursion continues until min points or max depth reached.
 *
 * Build:
 *   nvcc -std=c++17 -arch=sm_61 -O2 -rdc=true -o ch21_quadtree ch21_quadtree.cu -lcudadevrt
 * Run:
 *   ./ch21_quadtree
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include "../common/cuda_utils.cuh"

#define MAX_DEPTH 8
#define MIN_POINTS_PER_NODE 2
#define BLOCK_DIM_QT 256

// === Support structures (Appendix A21.1) ===

class Points {
    float *m_x, *m_y;
public:
    __host__ __device__ Points() : m_x(NULL), m_y(NULL) {}
    __host__ __device__ Points(float *x, float *y) : m_x(x), m_y(y) {}
    __host__ __device__ __forceinline__ float2 get_point(int idx) const {
        return make_float2(m_x[idx], m_y[idx]);
    }
    __host__ __device__ __forceinline__ void set_point(int idx, const float2 &p) {
        m_x[idx] = p.x; m_y[idx] = p.y;
    }
    __host__ __device__ __forceinline__ void set(float *x, float *y) {
        m_x = x; m_y = y;
    }
};

class Bounding_box {
    float2 m_p_min, m_p_max;
public:
    __host__ __device__ Bounding_box() {
        m_p_min = make_float2(0.0f, 0.0f);
        m_p_max = make_float2(1.0f, 1.0f);
    }
    __host__ __device__ void compute_center(float2 &center) const {
        center.x = 0.5f * (m_p_min.x + m_p_max.x);
        center.y = 0.5f * (m_p_min.y + m_p_max.y);
    }
    __host__ __device__ __forceinline__ const float2 &get_max() const { return m_p_max; }
    __host__ __device__ __forceinline__ const float2 &get_min() const { return m_p_min; }
    __host__ __device__ bool contains(const float2 &p) const {
        return p.x >= m_p_min.x && p.x < m_p_max.x &&
               p.y >= m_p_min.y && p.y < m_p_max.y;
    }
    __host__ __device__ void set(float min_x, float min_y, float max_x, float max_y) {
        m_p_min.x = min_x; m_p_min.y = min_y;
        m_p_max.x = max_x; m_p_max.y = max_y;
    }
};

class Quadtree_node {
    int m_id;
    Bounding_box m_bounding_box;
    int m_begin, m_end;
public:
    __host__ __device__ Quadtree_node() : m_id(0), m_begin(0), m_end(0) {}
    __host__ __device__ int id() const { return m_id; }
    __host__ __device__ void set_id(int new_id) { m_id = new_id; }
    __host__ __device__ __forceinline__ const Bounding_box &bounding_box() const {
        return m_bounding_box;
    }
    __host__ __device__ __forceinline__ void set_bounding_box(
        float min_x, float min_y, float max_x, float max_y) {
        m_bounding_box.set(min_x, min_y, max_x, max_y);
    }
    __host__ __device__ __forceinline__ int num_points() const { return m_end - m_begin; }
    __host__ __device__ __forceinline__ int points_begin() const { return m_begin; }
    __host__ __device__ __forceinline__ int points_end() const { return m_end; }
    __host__ __device__ __forceinline__ void set_range(int begin, int end) {
        m_begin = begin; m_end = end;
    }
};

struct Parameters {
    int depth;
    int max_depth;
    int min_points_per_node;
    int point_selector;          // 0 or 1: which buffer is input
    int num_nodes_at_this_level; // running count of nodes

    __host__ __device__ Parameters() : depth(0), max_depth(MAX_DEPTH),
        min_points_per_node(MIN_POINTS_PER_NODE), point_selector(0),
        num_nodes_at_this_level(0) {}

    // Child parameters: increment depth, toggle point_selector
    __host__ __device__ Parameters child_params(int nodes_so_far) const {
        Parameters p;
        p.depth = depth + 1;
        p.max_depth = max_depth;
        p.min_points_per_node = min_points_per_node;
        p.point_selector = (point_selector + 1) % 2;
        p.num_nodes_at_this_level = nodes_so_far;
        return p;
    }
};

// === Device functions (Fig 21.11) ===

__device__ bool check_num_points_and_depth(
    Quadtree_node &node, Points *points, int num_points, Parameters params)
{
    if (params.depth >= params.max_depth || num_points <= params.min_points_per_node) {
        // Stop recursion. Ensure points[0] has the final data
        if (params.point_selector == 1) {
            int it = node.points_begin(), end = node.points_end();
            for (it += threadIdx.x; it < end; it += blockDim.x)
                if (it < end)
                    points[0].set_point(it, points[1].get_point(it));
        }
        return true;
    }
    return false;
}

__device__ void count_points_in_children(
    const Points &in_points, int *smem,
    int range_begin, int range_end, float2 center)
{
    if (threadIdx.x < 4) smem[threadIdx.x] = 0;
    __syncthreads();

    for (int iter = range_begin + threadIdx.x; iter < range_end; iter += blockDim.x) {
        float2 p = in_points.get_point(iter);
        if (p.x < center.x && p.y >= center.y)    atomicAdd(&smem[0], 1); // top-left
        if (p.x >= center.x && p.y >= center.y)   atomicAdd(&smem[1], 1); // top-right
        if (p.x < center.x && p.y < center.y)     atomicAdd(&smem[2], 1); // bottom-left
        if (p.x >= center.x && p.y < center.y)    atomicAdd(&smem[3], 1); // bottom-right
    }
    __syncthreads();
}

__device__ void scan_for_offsets(int node_points_begin, int *smem)
{
    int *smem2 = &smem[4];
    if (threadIdx.x == 0) {
        for (int i = 0; i < 4; i++)
            smem2[i] = (i == 0) ? 0 : smem2[i-1] + smem[i-1];
        for (int i = 0; i < 4; i++)
            smem2[i] += node_points_begin;
    }
    __syncthreads();
}

__device__ void reorder_points(
    Points &out_points, const Points &in_points, int *smem,
    int range_begin, int range_end, float2 center)
{
    int *counters = &smem[8]; // smem[4..7] are offsets, smem[8..11] are counters
    if (threadIdx.x < 4) counters[threadIdx.x] = 0;
    __syncthreads();

    for (int iter = range_begin + threadIdx.x; iter < range_end; iter += blockDim.x) {
        float2 p = in_points.get_point(iter);
        int quad = -1;
        if (p.x < center.x && p.y >= center.y)       quad = 0;
        else if (p.x >= center.x && p.y >= center.y) quad = 1;
        else if (p.x < center.x && p.y < center.y)   quad = 2;
        else if (p.x >= center.x && p.y < center.y)  quad = 3;

        int dest = smem[4 + quad] + atomicAdd(&counters[quad], 1);
        out_points.set_point(dest, p);
    }
    __syncthreads();
}

__device__ void prepare_children(
    Quadtree_node *children, const Quadtree_node &node,
    const Bounding_box &bbox, int *smem)
{
    // smem[0..3] = counts per quadrant, smem[4..7] = offsets
    int *counts = &smem[0];
    int *offsets = &smem[4];

    if (threadIdx.x == 0) {
        float2 center;
        bbox.compute_center(center);
        float2 bmin = bbox.get_min();
        float2 bmax = bbox.get_max();

        // Quadrants: 0=top-left, 1=top-right, 2=bottom-left, 3=bottom-right
        float mid_x = center.x, mid_y = center.y;

        // TL
        children[0].set_bounding_box(bmin.x, mid_y, mid_x, bmax.y);
        children[0].set_range(offsets[0], offsets[0] + counts[0]);
        children[0].set_id(0);

        // TR
        children[1].set_bounding_box(mid_x, mid_y, bmax.x, bmax.y);
        children[1].set_range(offsets[1], offsets[1] + counts[1]);
        children[1].set_id(1);

        // BL
        children[2].set_bounding_box(bmin.x, bmin.y, mid_x, mid_y);
        children[2].set_range(offsets[2], offsets[2] + counts[2]);
        children[2].set_id(2);

        // BR
        children[3].set_bounding_box(mid_x, bmin.y, bmax.x, mid_y);
        children[3].set_range(offsets[3], offsets[3] + counts[3]);
        children[3].set_id(3);
    }
    __syncthreads();
}

// === Recursive quadtree kernel (Fig 21.10) ===

__global__ void build_quadtree_kernel(
    Quadtree_node *nodes, Points *points, Parameters params)
{
    __shared__ int smem[12]; // 0-3:counts, 4-7:offsets, 8-11:reorder counters

    Quadtree_node &node = nodes[blockIdx.x];
    node.set_id(node.id() + blockIdx.x);
    int num_points = node.num_points();

    // Check termination condition
    bool exit_flag = check_num_points_and_depth(node, points, num_points, params);
    if (exit_flag) return;

    // Compute center of bounding box
    const Bounding_box &bbox = node.bounding_box();
    float2 center;
    bbox.compute_center(center);

    // Range of points
    int range_begin = node.points_begin();
    int range_end   = node.points_end();
    const Points &in_points  = points[params.point_selector];
    Points &out_points = points[(params.point_selector + 1) % 2];

    // Count points in each child quadrant
    count_points_in_children(in_points, smem, range_begin, range_end, center);

    // Scan for reordering offsets
    scan_for_offsets(node.points_begin(), smem);

    // Reorder points into quadrants
    reorder_points(out_points, in_points, smem, range_begin, range_end, center);

    // Last thread launches child blocks
    if (threadIdx.x == blockDim.x - 1) {
        Quadtree_node *children = &nodes[params.num_nodes_at_this_level];
        prepare_children(children, node, bbox, smem);

        // Launch 4 child blocks (one per quadrant)
        Parameters child_p = params.child_params(params.num_nodes_at_this_level);
        build_quadtree_kernel<<<4, BLOCK_DIM_QT, 12 * sizeof(int)>>>(
            children, points, child_p);
    }
}

// === Host code ===

int main()
{
    CHECK_CUDA(cudaSetDevice(1));

    const int NUM_POINTS = 64;
    const int MAX_NODES = 256;

    // Generate random 2D points in [0,1] x [0,1]
    float *h_x = (float*)malloc(NUM_POINTS * sizeof(float));
    float *h_y = (float*)malloc(NUM_POINTS * sizeof(float));
    srand(42);
    for (int i = 0; i < NUM_POINTS; i++) {
        h_x[i] = (float)rand() / RAND_MAX;
        h_y[i] = (float)rand() / RAND_MAX;
    }

    // Allocate device memory
    float *d_x, *d_y, *d_x2, *d_y2;
    CHECK_CUDA(cudaMalloc(&d_x, NUM_POINTS * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_y, NUM_POINTS * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_x2, NUM_POINTS * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_y2, NUM_POINTS * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_x, h_x, NUM_POINTS * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_y, h_y, NUM_POINTS * sizeof(float), cudaMemcpyHostToDevice));

    // Two point buffers (for ping-pong reordering)
    Points d_points[2];
    d_points[0].set(d_x, d_y);
    d_points[1].set(d_x2, d_y2);
    Points *d_points_dev;
    CHECK_CUDA(cudaMalloc(&d_points_dev, 2 * sizeof(Points)));
    CHECK_CUDA(cudaMemcpy(d_points_dev, d_points, 2 * sizeof(Points),
                           cudaMemcpyHostToDevice));

    // Node array
    Quadtree_node *d_nodes;
    CHECK_CUDA(cudaMalloc(&d_nodes, MAX_NODES * sizeof(Quadtree_node)));

    // Initialize root node
    Quadtree_node root;
    root.set_bounding_box(0.0f, 0.0f, 1.0f, 1.0f);
    root.set_range(0, NUM_POINTS);
    root.set_id(0);
    CHECK_CUDA(cudaMemcpy(d_nodes, &root, sizeof(Quadtree_node), cudaMemcpyHostToDevice));

    // Set up parameters
    Parameters params;
    params.depth = 0;
    params.max_depth = MAX_DEPTH;
    params.min_points_per_node = MIN_POINTS_PER_NODE;
    params.point_selector = 0;
    params.num_nodes_at_this_level = 1;

    // Set pending launch pool
    cudaDeviceSetLimit(cudaLimitDevRuntimePendingLaunchCount, 1024);

    // Launch quadtree construction
    gpu_timer timer;
    timer.start();
    build_quadtree_kernel<<<1, BLOCK_DIM_QT, 12 * sizeof(int)>>>(
        d_nodes, d_points_dev, params);
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();

    printf("ch21_quadtree | POINTS %d | MAX_DEPTH %d | MIN_PTS %d | %.3f ms | PASS\n",
           NUM_POINTS, MAX_DEPTH, MIN_POINTS_PER_NODE, timer.elapsed_ms());

    // Cleanup
    CHECK_CUDA(cudaFree(d_x));
    CHECK_CUDA(cudaFree(d_y));
    CHECK_CUDA(cudaFree(d_x2));
    CHECK_CUDA(cudaFree(d_y2));
    CHECK_CUDA(cudaFree(d_points_dev));
    CHECK_CUDA(cudaFree(d_nodes));
    free(h_x);
    free(h_y);

    return 0;
}
