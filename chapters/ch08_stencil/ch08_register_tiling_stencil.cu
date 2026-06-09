/*
 * Chapter 8: Stencil
 * Register Tiling for Stencil Sweep — Fig 8.12
 *
 * Extends the thread coarsening kernel (Fig 8.10) by moving z-neighbor
 * values into registers instead of shared memory.
 *
 * Key insight: for a 3D seven-point stencil, the z-neighbors (inPrev, inNext)
 * are each accessed by exactly ONE thread. Only the x-y neighbors (inCurr_s)
 * are shared across threads in the same plane.
 *
 * Benefits vs Fig 8.10:
 *   - Shared memory reduced by 2/3 (only 1 plane instead of 3)
 *   - Lower-latency register access for 2/3 of input load
 *   - Trade-off: 2 extra registers per thread
 *
 * Structure per z-plane:
 *   inPrev  (register):  z-1 neighbor (this thread's x,y only)
 *   inCurr_s(shared):    current z-plane (full tile, all threads)
 *   inNext  (register):  z+1 neighbor (this thread's x,y only)
 *
 * Compute: GTX 1050, sm_61 (device 1)
 */

#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstring>
#include "../common/cuda_utils.cuh"

/* Stencil coefficients */
#define COEFF_CENTER  -6.0f
#define COEFF_XM1      1.0f
#define COEFF_XP1      1.0f
#define COEFF_YM1      1.0f
#define COEFF_YP1      1.0f
#define COEFF_ZM1      1.0f
#define COEFF_ZP1      1.0f

/* Tile configuration */
#define TILE_DIM   32  /* output tile x-y dimension */
#define HALO        1  /* order 1 stencil */
#define BLOCK_DIM  TILE_DIM

/*
 * Register-tiled 3D seven-point stencil kernel (Fig 8.12).
 *
 * Thread block is TILE_DIM × TILE_DIM (2D). Each thread processes
 * a column of output grid points in z, as in Fig 8.10.
 *
 * However, only the current z-plane is stored in shared memory (inCurr_s).
 * The z-1 and z+1 planes are kept in per-thread registers (inPrev, inNext)
 * because they are each read by only one thread.
 *
 * Parameters:
 *   in   - 3D input grid (N x N x N, row-major)
 *   out  - 3D output grid (N x N x N, row-major)
 *   N    - grid dimension in each axis
 */
__global__ void register_tiling_stencil_kernel(
    const float* __restrict__ in,
    float* __restrict__ out,
    int N)
{
    // Shared memory: only the CURRENT z-plane (x-y neighbors need sharing)
    __shared__ float inCurr_s[BLOCK_DIM + 2][BLOCK_DIM + 2];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Global x, y coordinates for this thread
    int i = blockIdx.x * TILE_DIM + tx;
    int j = blockIdx.y * TILE_DIM + ty;

    // Output z range
    int z_start = blockIdx.z * (TILE_DIM - 2) + 1;
    int z_end   = min(z_start + (TILE_DIM - 2) - 1, N - 2);

    int num_planes = z_end - z_start + 1;
    if (num_planes <= 0) return;

    // Per-thread registers for z-neighbors
    float inPrev;  // holds z-1 plane value at this (i,j)
    float inNext;  // holds z+1 plane value at this (i,j)

    // ------------------------------------------------------------------
    // Phase 1: Initial load
    // ------------------------------------------------------------------
    // Load inCurr_s (current plane at z_start) — cooperative shared memory fill
    for (int offset = 0; offset < (BLOCK_DIM + 2) * (BLOCK_DIM + 2);
         offset += blockDim.x * blockDim.y)
    {
        int flat = tx + ty * blockDim.x + offset;
        int lx = flat % (BLOCK_DIM + 2);
        int ly = flat / (BLOCK_DIM + 2);
        if (ly < BLOCK_DIM + 2) {
            int gi = blockIdx.x * TILE_DIM + lx - HALO;
            int gj = blockIdx.y * TILE_DIM + ly - HALO;
            if (gi >= 0 && gi < N && gj >= 0 && gj < N && z_start < N)
                inCurr_s[ly][lx] = in[z_start * N * N + gj * N + gi];
            else
                inCurr_s[ly][lx] = 0.0f;
        }
    }

    // Load inPrev (z-1 plane at z_start-1) — per-thread register load
    int z_prev = z_start - 1;
    if (tx < TILE_DIM && ty < TILE_DIM && i >= 0 && i < N && j >= 0 && j < N &&
        z_prev >= 0 && z_prev < N)
        inPrev = in[z_prev * N * N + j * N + i];
    else
        inPrev = 0.0f;

    __syncthreads();

    // ------------------------------------------------------------------
    // Phase 2: Compute output planes iteratively
    // ------------------------------------------------------------------
    for (int z_plane = z_start; z_plane <= z_end; ++z_plane)
    {
        // Load inNext (z+1 plane at z_plane+1) — per-thread register load
        int z_next = z_plane + 1;
        if (tx < TILE_DIM && ty < TILE_DIM && i >= 0 && i < N && j >= 0 && j < N &&
            z_next >= 0 && z_next < N)
            inNext = in[z_next * N * N + j * N + i];
        else
            inNext = 0.0f;

        __syncthreads();

        // Compute output for this plane
        if (i >= 1 && i < N-1 && j >= 1 && j < N-1 &&
            z_plane >= 1 && z_plane < N-1 &&
            tx < TILE_DIM && ty < TILE_DIM)
        {
            int sx = tx + HALO;
            int sy = ty + HALO;

            float c  = inCurr_s[sy][sx];
            float xm = inCurr_s[sy][sx - 1];
            float xp = inCurr_s[sy][sx + 1];
            float ym = inCurr_s[sy - 1][sx];
            float yp = inCurr_s[sy + 1][sx];

            int out_idx = z_plane * N * N + j * N + i;
            out[out_idx] = COEFF_CENTER * c
                         + COEFF_XM1    * xm
                         + COEFF_XP1    * xp
                         + COEFF_YM1    * ym
                         + COEFF_YP1    * yp
                         + COEFF_ZM1    * inPrev
                         + COEFF_ZP1    * inNext;
        }

        __syncthreads();

        // ------------------------------------------------------------------
        // Phase 3: Roll for next iteration
        //   inPrev (reg) ← inCurr_s[ty+HALO][tx+HALO] (current → z-1 for next)
        //   Full cooperative reload of inCurr_s from global (next z-plane)
        // ------------------------------------------------------------------

        // Save the old current plane value to inPrev register
        if (tx < TILE_DIM && ty < TILE_DIM)
            inPrev = inCurr_s[ty + HALO][tx + HALO];

        // Cooperative reload of inCurr_s with the next z-plane (z_plane + 1)
        // This ensures halo cells are properly loaded for the next iteration
        int z_new = z_plane + 1;
        for (int offset = 0; offset < (BLOCK_DIM + 2) * (BLOCK_DIM + 2);
             offset += blockDim.x * blockDim.y)
        {
            int flat = tx + ty * blockDim.x + offset;
            int lx = flat % (BLOCK_DIM + 2);
            int ly = flat / (BLOCK_DIM + 2);
            if (ly < BLOCK_DIM + 2) {
                int gi = blockIdx.x * TILE_DIM + lx - HALO;
                int gj = blockIdx.y * TILE_DIM + ly - HALO;
                if (gi >= 0 && gi < N && gj >= 0 && gj < N && z_new >= 0 && z_new < N)
                    inCurr_s[ly][lx] = in[z_new * N * N + gj * N + gi];
                else
                    inCurr_s[ly][lx] = 0.0f;
            }
        }

        __syncthreads();
    }
}

/* -------------------------------------------------------------------------- */
/*  CPU reference                                                             */
/* -------------------------------------------------------------------------- */
void cpu_stencil_sweep(const float* in, float* out, int N)
{
    for (int k = 1; k < N-1; ++k)
        for (int j = 1; j < N-1; ++j)
            for (int i = 1; i < N-1; ++i) {
                int idx = k * N * N + j * N + i;
                out[idx] = COEFF_CENTER * in[idx]
                         + COEFF_XM1    * in[idx - 1]
                         + COEFF_XP1    * in[idx + 1]
                         + COEFF_YM1    * in[idx - N]
                         + COEFF_YP1    * in[idx + N]
                         + COEFF_ZM1    * in[idx - N*N]
                         + COEFF_ZP1    * in[idx + N*N];
            }
}

/* -------------------------------------------------------------------------- */
/*  Main                                                                      */
/* -------------------------------------------------------------------------- */
int main()
{
    cudaSetDevice(1);
    CHECK_CUDA(cudaGetLastError());

    const int N = 64;
    const int total = N * N * N;

    std::cout << "====================================================\n";
    std::cout << "Chapter 8: Register Tiling Stencil (Fig 8.12)\n";
    std::cout << "====================================================\n";
    std::cout << "Grid: " << N << " x " << N << " x " << N << "\n";
    std::cout << "Block (2D): " << BLOCK_DIM << " x " << BLOCK_DIM << "\n";
    std::cout << "Shared mem (1 plane): "
              << ((BLOCK_DIM+2)*(BLOCK_DIM+2)*sizeof(float)) << " B = "
              << ((BLOCK_DIM+2)*(BLOCK_DIM+2)*sizeof(float)/1024) << " KB\n";
    std::cout << "Registers per thread: +2 (inPrev, inNext)\n";
    std::cout << "Device: GTX 1050 (sm_61)\n\n";

    // Host memory
    float* h_in  = new float[total];
    float* h_out = new float[total];
    float* h_ref = new float[total];

    // Initialize
    for (int k = 0; k < N; ++k)
        for (int j = 0; j < N; ++j)
            for (int i = 0; i < N; ++i)
                h_in[k * N * N + j * N + i] = sinf(i * 0.1f) * cosf(j * 0.1f) * sinf(k * 0.1f);

    // CPU reference
    memcpy(h_ref, h_in, total * sizeof(float));
    cpu_stencil_sweep(h_in, h_ref, N);

    // Device memory
    float *d_in, *d_out;
    CHECK_CUDA(cudaMalloc(&d_in,  total * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_out, total * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_in,  h_in,  total * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_out, h_in, total * sizeof(float), cudaMemcpyHostToDevice));

    // Grid
    dim3 blockDim(BLOCK_DIM, BLOCK_DIM);
    dim3 gridDim(
        (N + BLOCK_DIM - 1) / BLOCK_DIM,
        (N + BLOCK_DIM - 1) / BLOCK_DIM,
        ((N - 2) + (TILE_DIM - 2) - 1) / (TILE_DIM - 2)
    );

    std::cout << "Grid:  " << gridDim.x << " x " << gridDim.y << " x " << gridDim.z << "\n";
    std::cout << "Blocks: " << gridDim.x * gridDim.y * gridDim.z << "\n\n";

    // Warm-up
    register_tiling_stencil_kernel<<<gridDim, blockDim>>>(d_in, d_out, N);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Timed run
    gpu_timer timer;
    timer.start();
    register_tiling_stencil_kernel<<<gridDim, blockDim>>>(d_in, d_out, N);
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();

    float kernel_time = timer.elapsed_ms();

    // Copy result back
    CHECK_CUDA(cudaMemcpy(h_out, d_out, total * sizeof(float), cudaMemcpyDeviceToHost));

    // Metrics
    int interior = (N-2) * (N-2) * (N-2);
    float total_ops = interior * 13.0f;
    float gflops = total_ops / (kernel_time * 1e6f);

    // Validate
    bool valid = cpu_allclose(h_ref, h_out, total);

    std::cout << "Results:\n";
    std::cout << "  Kernel time:     " << std::fixed << std::setprecision(3) << kernel_time << " ms\n";
    std::cout << "  Interior cells:  " << interior << "\n";
    std::cout << "  GFLOPS:          " << std::fixed << std::setprecision(2) << gflops << "\n";
    std::cout << "  Validation:      " << (valid ? "PASSED" : "FAILED") << "\n\n";

    // Cleanup
    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_out));
    delete[] h_in;
    delete[] h_out;
    delete[] h_ref;

    return valid ? 0 : 1;
}
