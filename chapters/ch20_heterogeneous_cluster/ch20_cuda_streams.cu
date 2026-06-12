/*
 * ch20_cuda_streams.cu — CUDA Streams + Pinned Memory Stencil Demo
 *
 * From PMPP 4th Ed, Chapter 20: Programming a Heterogeneous Computing Cluster
 *
 * Demonstrates the CUDA-specific concepts from Ch20 without requiring MPI:
 *   1. Pinned (page-locked) host memory via cudaHostAlloc()
 *   2. CUDA streams for concurrent kernel execution
 *   3. Asynchronous memory copies (cudaMemcpyAsync)
 *   4. Overlap of computation on different grid regions
 *
 * The demo splits a 3D stencil grid into two halves, processes each half in a
 * separate stream, and compares single-stream vs dual-stream performance.
 *
 * Note: The full MPI+CUDA distributed stencil requires MPI (OpenMPI/MPICH).
 * This demo focuses on the CUDA-side mechanisms used in that application.
 *
 * Build: nvcc -std=c++17 -arch=sm_61 -O2 -o ch20_cuda_streams ch20_cuda_streams.cu
 * Run:   ./ch20_cuda_streams
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include "../common/cuda_utils.cuh"

// 3D 7-point stencil (same as Ch8 basic stencil)
// Each point averages its 6 axial neighbors
__global__ void stencil_3d_kernel(const float *__restrict__ in, float *__restrict__ out,
                                   int dimx, int dimy, int dimz,
                                   int z_start, int z_end)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = z_start + blockIdx.z * blockDim.z + threadIdx.z;

    if (i < 1 || i >= dimx - 1 || j < 1 || j >= dimy - 1 || k >= z_end) return;

    int idx = (k * dimy + j) * dimx + i;

    out[idx] = (in[idx] * 0.5f +
                in[idx - 1] * 0.1f + in[idx + 1] * 0.1f +
                in[idx - dimx] * 0.1f + in[idx + dimx] * 0.1f +
                in[idx - dimx * dimy] * 0.1f + in[idx + dimx * dimy] * 0.1f);
}

// CPU reference
void cpu_stencil(const float *in, float *out, int dimx, int dimy, int dimz)
{
    for (int k = 1; k < dimz - 1; k++) {
        for (int j = 1; j < dimy - 1; j++) {
            for (int i = 1; i < dimx - 1; i++) {
                int idx = (k * dimy + j) * dimx + i;
                out[idx] = (in[idx] * 0.5f +
                           in[idx - 1] * 0.1f + in[idx + 1] * 0.1f +
                           in[idx - dimx] * 0.1f + in[idx + dimx] * 0.1f +
                           in[idx - dimx * dimy] * 0.1f + in[idx + dimx * dimy] * 0.1f);
            }
        }
    }
}

int main()
{
    CHECK_CUDA(cudaSetDevice(1));

    // Grid dimensions
    const int DIMX = 64;
    const int DIMY = 64;
    const int DIMZ = 64;
    const int N = DIMX * DIMY * DIMZ;
    size_t bytes = N * sizeof(float);

    // Allocate pinned host memory (same as cudaHostAlloc in Ch20 Fig 20.11)
    float *h_input, *h_output, *h_ref;
    CHECK_CUDA(cudaHostAlloc((void**)&h_input, bytes, cudaHostAllocDefault));
    CHECK_CUDA(cudaHostAlloc((void**)&h_output, bytes, cudaHostAllocDefault));
    CHECK_CUDA(cudaHostAlloc((void**)&h_ref, bytes, cudaHostAllocDefault));

    // Initialize with random data
    srand(42);
    for (int i = 0; i < N; i++)
        h_input[i] = (float)rand() / RAND_MAX;

    // CPU reference
    cpu_stencil(h_input, h_ref, DIMX, DIMY, DIMZ);

    // Device memory
    float *d_input, *d_output;
    CHECK_CUDA(cudaMalloc(&d_input, bytes));
    CHECK_CUDA(cudaMalloc(&d_output, bytes));

    // Copy input to device
    CHECK_CUDA(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));

    // ============================================
    // TEST 1: Single stream (baseline)
    // ============================================
    dim3 block_dim(8, 8, 4);
    dim3 grid_dim((DIMX + 7) / 8, (DIMY + 7) / 8, (DIMZ + 3) / 4);

    gpu_timer t1;
    t1.start();
    stencil_3d_kernel<<<grid_dim, block_dim>>>(d_input, d_output, DIMX, DIMY, DIMZ, 0, DIMZ);
    t1.stop();

    // ============================================
    // TEST 2: Two streams (concurrent halves)
    // ============================================
    cudaStream_t stream0, stream1;
    CHECK_CUDA(cudaStreamCreate(&stream0));
    CHECK_CUDA(cudaStreamCreate(&stream1));

    int z_mid = DIMZ / 2;
    int z_half_grids = (z_mid + 3) / 4 + 1;  // +1 for overlap
    dim3 half_grid((DIMX + 7) / 8, (DIMY + 7) / 8, z_half_grids);

    gpu_timer t2;
    t2.start();

    // Launch top half in stream0
    stencil_3d_kernel<<<half_grid, block_dim, 0, stream0>>>(
        d_input, d_output, DIMX, DIMY, DIMZ, 0, z_mid + 1);

    // Launch bottom half in stream1 (concurrent with stream0)
    stencil_3d_kernel<<<half_grid, block_dim, 0, stream1>>>(
        d_input, d_output, DIMX, DIMY, DIMZ, z_mid - 1, DIMZ);

    // Wait for both streams to complete
    CHECK_CUDA(cudaStreamSynchronize(stream0));
    CHECK_CUDA(cudaStreamSynchronize(stream1));
    t2.stop();

    // ============================================
    // TEST 3: Async copy with kernel overlap
    // ============================================
    // This demonstrates the pattern in Ch20 Fig 20.13-20.15:
    //   stream0: compute boundary + D2H copy
    //   stream1: compute interior concurrently
    //
    // For the demo: stream0 does top half and copies result out,
    // while stream1 does bottom half concurrently

    float *h_pinned_buf;  // bounce buffer (like Ch20 h_left_boundary)
    CHECK_CUDA(cudaHostAlloc((void**)&h_pinned_buf, z_mid * DIMY * DIMX * sizeof(float),
                              cudaHostAllocDefault));

    // Reset device output
    CHECK_CUDA(cudaMemset(d_output, 0, bytes));

    gpu_timer t3;
    t3.start();

    // Stream0: compute top half, then async copy result to pinned host memory
    stencil_3d_kernel<<<half_grid, block_dim, 0, stream0>>>(
        d_input, d_output, DIMX, DIMY, DIMZ, 0, z_mid + 1);
    CHECK_CUDA(cudaMemcpyAsync(h_pinned_buf, d_output,
                                z_mid * DIMY * DIMX * sizeof(float),
                                cudaMemcpyDeviceToHost, stream0));

    // Stream1: compute bottom half concurrently with stream0's copy
    stencil_3d_kernel<<<half_grid, block_dim, 0, stream1>>>(
        d_input, d_output, DIMX, DIMY, DIMZ, z_mid - 1, DIMZ);

    CHECK_CUDA(cudaDeviceSynchronize());
    t3.stop();

    // ============================================
    // Validation
    // ============================================
    CHECK_CUDA(cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost));

    bool pass = true;
    int fail_count = 0;
    for (int i = 1; i < DIMX - 1 && pass; i++) {
        for (int j = 1; j < DIMY - 1 && pass; j++) {
            for (int k = 1; k < DIMZ - 1 && pass; k++) {
                int idx = (k * DIMY + j) * DIMX + i;
                float diff = fabsf(h_output[idx] - h_ref[idx]);
                float denom = fmaxf(fabsf(h_ref[idx]), 1e-10f);
                if (diff / denom > 1e-4f && diff > 1e-6f) {
                    printf("FAIL at [%d,%d,%d]: gpu=%.6e ref=%.6e diff=%.2e\n",
                           i, j, k, h_output[idx], h_ref[idx], diff);
                    fail_count++;
                    if (fail_count >= 5) { pass = false; break; }
                }
            }
        }
    }

    printf("ch20_cuda_streams | GRID %dx%dx%d | Single %.3f ms | Dual %.3f ms | Async %.3f ms | %s\n",
           DIMX, DIMY, DIMZ, t1.elapsed_ms(), t2.elapsed_ms(), t3.elapsed_ms(),
           pass ? "PASS" : "FAIL");

    // Cleanup
    CHECK_CUDA(cudaStreamDestroy(stream0));
    CHECK_CUDA(cudaStreamDestroy(stream1));
    CHECK_CUDA(cudaFreeHost(h_input));
    CHECK_CUDA(cudaFreeHost(h_output));
    CHECK_CUDA(cudaFreeHost(h_ref));
    CHECK_CUDA(cudaFreeHost(h_pinned_buf));
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));

    return pass ? 0 : 1;
}
