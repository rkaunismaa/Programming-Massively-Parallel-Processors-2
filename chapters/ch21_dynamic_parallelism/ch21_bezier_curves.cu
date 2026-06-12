/*
 * ch21_bezier_curves.cu — Bezier Curves with CUDA Dynamic Parallelism
 *
 * From PMPP 4th Ed, Chapter 21: CUDA Dynamic Parallelism
 * Fig 21.7: Parent launches child kernel for each Bezier line.
 *
 * Key DP features demonstrated:
 *   1. Parent kernel discovers work amount, launches child grid
 *   2. cudaMalloc from device code
 *   3. cudaFree from device code (freeVertexMem kernel)
 *   4. Child kernels run in parallel (asynchronous launch)
 *
 * Note: Device-allocated memory (cudaMalloc from a kernel) is device-only
 * and cannot be directly accessed via host cudaMemcpy on sm_61 GPUs.
 * Validation focuses on vertex counts and error-free execution.
 *
 * Build:
 *   nvcc -std=c++17 -arch=sm_61 -O2 -rdc=true -o ch21_bezier_curves ch21_bezier_curves.cu -lcudadevrt
 * Run:
 *   ./ch21_bezier_curves
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include "../common/cuda_utils.cuh"

#define MAX_TESS_POINTS 32
#define BLOCK_DIM 64

struct BezierLine {
    float2 CP[3];
    float2 *vertexPos;       // device-allocated (DP) or host-allocated
    int nVertices;
};

// Compute curvature for a Bezier line
__device__ float computeCurvature(const BezierLine *bLines, int lidx)
{
    float2 p0 = bLines[lidx].CP[0];
    float2 p1 = bLines[lidx].CP[1];
    float2 p2 = bLines[lidx].CP[2];
    float chord_len = sqrtf((p2.x - p0.x)*(p2.x - p0.x) +
                            (p2.y - p0.y)*(p2.y - p0.y));
    if (chord_len < 1e-6f) return 0.0f;
    float cross = fabsf((p2.x - p0.x)*(p0.y - p1.y) -
                         (p0.x - p1.x)*(p2.y - p0.y));
    return cross / chord_len;
}

// === Dynamic Parallelism kernels (Fig 21.7) ===

// Child kernel (must be defined BEFORE parent for DP)
__global__ void computeBezierLine_child(int lidx, BezierLine *bLines, int nTessPoints)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < nTessPoints) {
        float u = (float)idx / (float)(nTessPoints - 1);
        float omu = 1.0f - u;
        float B3u[3];
        B3u[0] = omu * omu;
        B3u[1] = 2.0f * u * omu;
        B3u[2] = u * u;
        float2 position = make_float2(0.0f, 0.0f);
        for (int i = 0; i < 3; i++) {
            position.x += B3u[i] * bLines[lidx].CP[i].x;
            position.y += B3u[i] * bLines[lidx].CP[i].y;
        }
        bLines[lidx].vertexPos[idx] = position;
    }
}

// Parent kernel: discovers work, allocates memory, launches child
__global__ void computeBezierLines_parent(BezierLine *bLines, int nLines)
{
    int lidx = threadIdx.x + blockDim.x * blockIdx.x;
    if (lidx < nLines) {
        float curvature = computeCurvature(bLines, lidx);
        bLines[lidx].nVertices = min(max((int)(curvature * 16.0f), 4), MAX_TESS_POINTS);

        cudaMalloc((void**)&bLines[lidx].vertexPos,
                    bLines[lidx].nVertices * sizeof(float2));

        int blocks = (bLines[lidx].nVertices + 31) / 32;
        computeBezierLine_child<<<blocks, 32>>>(lidx, bLines, bLines[lidx].nVertices);
    }
}

// Free kernel: device memory must be freed by a device kernel
__global__ void freeVertexMem(BezierLine *bLines, int nLines)
{
    int lidx = threadIdx.x + blockDim.x * blockIdx.x;
    if (lidx < nLines && bLines[lidx].vertexPos != NULL)
        cudaFree(bLines[lidx].vertexPos);
}

// CPU reference
void cpu_bezier_counts(float2 CP[3], int *nTessPoints)
{
    float2 p0 = CP[0], p1 = CP[1], p2 = CP[2];
    float chord_len = sqrtf((p2.x-p0.x)*(p2.x-p0.x) + (p2.y-p0.y)*(p2.y-p0.y));
    float curvature = 0.0f;
    if (chord_len > 1e-6f)
        curvature = fabsf((p2.x-p0.x)*(p0.y-p1.y) - (p0.x-p1.x)*(p2.y-p0.y)) / chord_len;
    int n = (int)(curvature * 16.0f);
    if (n < 4) n = 4;
    if (n > MAX_TESS_POINTS) n = MAX_TESS_POINTS;
    *nTessPoints = n;
}

int main()
{
    CHECK_CUDA(cudaSetDevice(1));

    const int N_LINES = 16;

    // Generate random control points
    BezierLine *h_lines = (BezierLine*)malloc(N_LINES * sizeof(BezierLine));
    srand(42);
    for (int i = 0; i < N_LINES; i++) {
        h_lines[i].CP[0] = make_float2((float)rand()/RAND_MAX, (float)rand()/RAND_MAX);
        h_lines[i].CP[1] = make_float2((float)rand()/RAND_MAX, (float)rand()/RAND_MAX);
        h_lines[i].CP[2] = make_float2((float)rand()/RAND_MAX, (float)rand()/RAND_MAX);
        h_lines[i].vertexPos = NULL;
        h_lines[i].nVertices = 0;
    }

    // CPU reference counts
    int cpu_counts[N_LINES];
    for (int i = 0; i < N_LINES; i++)
        cpu_bezier_counts(h_lines[i].CP, &cpu_counts[i]);

    // === TEST: Dynamic Parallelism ===
    BezierLine *d_lines;
    CHECK_CUDA(cudaMalloc(&d_lines, N_LINES * sizeof(BezierLine)));
    CHECK_CUDA(cudaMemcpy(d_lines, h_lines, N_LINES * sizeof(BezierLine),
                           cudaMemcpyHostToDevice));

    cudaDeviceSetLimit(cudaLimitDevRuntimePendingLaunchCount, N_LINES);

    int grid_blocks = (N_LINES + BLOCK_DIM - 1) / BLOCK_DIM;

    gpu_timer timer;
    timer.start();
    computeBezierLines_parent<<<grid_blocks, BLOCK_DIM>>>(d_lines, N_LINES);

    // Wait for parent and all child grids
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();

    // Copy back structs to inspect vertex counts
    CHECK_CUDA(cudaMemcpy(h_lines, d_lines, N_LINES * sizeof(BezierLine),
                           cudaMemcpyDeviceToHost));

    bool pass = true;
    for (int i = 0; i < N_LINES; i++) {
        if (h_lines[i].nVertices != cpu_counts[i]) {
            printf("FAIL line %d: gpu=%d cpu=%d\n", i, h_lines[i].nVertices, cpu_counts[i]);
            pass = false;
        }
    }

    // Free device memory (must be done from device kernel)
    freeVertexMem<<<grid_blocks, BLOCK_DIM>>>(d_lines, N_LINES);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaFree(d_lines));

    printf("ch21_bezier | LINES %d | %.3f ms | %s\n",
           N_LINES, timer.elapsed_ms(), pass ? "PASS" : "FAIL");

    free(h_lines);
    return pass ? 0 : 1;
}
