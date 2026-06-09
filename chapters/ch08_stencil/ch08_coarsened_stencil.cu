/*
 * Chapter 8: Stencil
 * Thread Coarsening for Stencil Sweep — Fig 8.10
 *
 * Implements a 3D seven-point stencil sweep with thread coarsening
 * in the z-direction. Instead of a 3D thread block (T³), uses a 2D
 * block (T²) where each thread processes an entire column in z.
 *
 * Key advantages:
 * - Block size is T² instead of T³ → T can be much larger (32×32 = 1024)
 * - Only 3T² shared memory needed instead of T³ (for T=32: 12 KB vs 128 KB)
 * - Much better memory coalescing (warp accesses contiguous columns)
 * - Higher arithmetic intensity: 2.68 OP/B at T=32 vs 1.37 OP/B at T=8
 *
 * Algorithm:
 *   1. Each block loads 3 z-planes of the input tile into shared memory
 *      (inPrev_s, inCurr_s, inNext_s)
 *   2. All threads compute one output plane from the 3 shared planes
 *   3. Roll shared memory: inPrev ← inCurr, inCurr ← inNext
 *   4. Load the next z-plane into inNext
 *   5. Repeat until all output planes in the tile are computed
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
#define TILE_DIM   32  /* output tile x-y dimension (block = TILE_DIM × TILE_DIM) */
#define HALO        1  /* order 1 stencil */
#define BLOCK_DIM  TILE_DIM  /* 2D block — same as TILE_DIM */
#define STRIDE_Z    1  /* z-stride for coarsening */

/*
 * Coarsened 3D seven-point stencil kernel (Fig 8.10).
 *
 * Thread block is 2D (TILE_DIM × TILE_DIM). Each thread processes
 * a column of output grid points in the z-direction.
 *
 * Shared memory holds 3 planes of the input tile:
 *   inPrev_s: previous z-plane (z-1 neighbor)
 *   inCurr_s: current z-plane (x-y neighbors)
 *   inNext_s: next z-plane (z+1 neighbor)
 *
 * Parameters:
 *   in   - 3D input grid (N x N x N, row-major)
 *   out  - 3D output grid (N x N x N, row-major)
 *   N    - grid dimension in each axis
 */
__global__ void coarsened_stencil_kernel(
    const float* __restrict__ in,
    float* __restrict__ out,
    int N)
{
    // Shared memory: 3 x-y planes of the input tile
    // Each plane is (BLOCK_DIM + 2*HALO) × (BLOCK_DIM + 2*HALO) = 34×34 = 1156 elements
    __shared__ float inPrev_s[BLOCK_DIM + 2][BLOCK_DIM + 2];
    __shared__ float inCurr_s[BLOCK_DIM + 2][BLOCK_DIM + 2];
    __shared__ float inNext_s[BLOCK_DIM + 2][BLOCK_DIM + 2];

    int tx = threadIdx.x;  // x within block (0..TILE_DIM-1)
    int ty = threadIdx.y;  // y within block (0..TILE_DIM-1)

    // Absolute output grid coordinates for this thread's x-y position
    int i = blockIdx.x * TILE_DIM + tx;
    int j = blockIdx.y * TILE_DIM + ty;

    // Output z range for this block (z is the coarsened dimension)
    // Each block processes (TILE_DIM - 2) output planes in z
    // Interior z planes are 1 .. N-2
    int z_start = blockIdx.z * (TILE_DIM - 2) + 1;
    int z_end   = min(z_start + (TILE_DIM - 2) - 1, N - 2);

    // Number of output planes this block will compute
    int num_planes = z_end - z_start + 1;
    if (num_planes <= 0) return;

    // -----------------------------------------------------------------------
    // Phase 1: Initial load — bring first two planes into shared memory
    // -----------------------------------------------------------------------

    // Load inPrev_s: the plane at z = z_start - 1 (z-1 neighbor for first output)
    for (int offset = 0; offset < (BLOCK_DIM + 2) * (BLOCK_DIM + 2);
         offset += blockDim.x * blockDim.y)
    {
        int flat = tx + ty * blockDim.x + offset;
        int lx = flat % (BLOCK_DIM + 2);
        int ly = flat / (BLOCK_DIM + 2);
        if (ly < BLOCK_DIM + 2) {
            int gi = blockIdx.x * TILE_DIM + lx - HALO;
            int gj = blockIdx.y * TILE_DIM + ly - HALO;
            if (gi >= 0 && gi < N && gj >= 0 && gj < N && (z_start - 1) >= 0 && (z_start - 1) < N)
                inPrev_s[ly][lx] = in[(z_start - 1) * N * N + gj * N + gi];
            else
                inPrev_s[ly][lx] = 0.0f;
        }
    }

    // Load inCurr_s: the plane at z = z_start (first output plane's z)
    for (int offset = 0; offset < (BLOCK_DIM + 2) * (BLOCK_DIM + 2);
         offset += blockDim.x * blockDim.y)
    {
        int flat = tx + ty * blockDim.x + offset;
        int lx = flat % (BLOCK_DIM + 2);
        int ly = flat / (BLOCK_DIM + 2);
        if (ly < BLOCK_DIM + 2) {
            int gi = blockIdx.x * TILE_DIM + lx - HALO;
            int gj = blockIdx.y * TILE_DIM + ly - HALO;
            if (gi >= 0 && gi < N && gj >= 0 && gj < N && z_start >= 0 && z_start < N)
                inCurr_s[ly][lx] = in[z_start * N * N + gj * N + gi];
            else
                inCurr_s[ly][lx] = 0.0f;
        }
    }

    // Iterate over z-planes
    for (int z_plane = z_start; z_plane <= z_end; ++z_plane)
    {
        // Load the next plane (z_plane + 1) into inNext_s
        for (int offset = 0; offset < (BLOCK_DIM + 2) * (BLOCK_DIM + 2);
             offset += blockDim.x * blockDim.y)
        {
            int flat = tx + ty * blockDim.x + offset;
            int lx = flat % (BLOCK_DIM + 2);
            int ly = flat / (BLOCK_DIM + 2);
            if (ly < BLOCK_DIM + 2) {
                int gi = blockIdx.x * TILE_DIM + lx - HALO;
                int gj = blockIdx.y * TILE_DIM + ly - HALO;
                if (gi >= 0 && gi < N && gj >= 0 && gj < N && (z_plane + 1) >= 0 && (z_plane + 1) < N)
                    inNext_s[ly][lx] = in[(z_plane + 1) * N * N + gj * N + gi];
                else
                    inNext_s[ly][lx] = 0.0f;
            }
        }

        __syncthreads();

        // Compute output only for interior cells within TILE_DIM in x,y
        if (i >= 1 && i < N-1 && j >= 1 && j < N-1 &&
            z_plane >= 1 && z_plane < N-1 &&
            tx < TILE_DIM && ty < TILE_DIM)
        {
            // Shared memory indices (offset by HALO for boundary halo)
            // inCurr_s[ly+1][lx+1]   = center (after halo offset)
            // inCurr_s[ly+1][lx]     = x-1
            // inCurr_s[ly+1][lx+2]   = x+1
            // inCurr_s[ly][lx+1]     = y-1
            // inCurr_s[ly+2][lx+1]   = y+1
            // inPrev_s[ly+1][lx+1]   = z-1
            // inNext_s[ly+1][lx+1]   = z+1

            int sx = tx + HALO;  // shared memory x index
            int sy = ty + HALO;  // shared memory y index

            float c  = inCurr_s[sy][sx];
            float xm = inCurr_s[sy][sx - 1];
            float xp = inCurr_s[sy][sx + 1];
            float ym = inCurr_s[sy - 1][sx];
            float yp = inCurr_s[sy + 1][sx];
            float zm = inPrev_s[sy][sx];
            float zp = inNext_s[sy][sx];

            int out_idx = z_plane * N * N + j * N + i;
            out[out_idx] = COEFF_CENTER * c
                         + COEFF_XM1    * xm
                         + COEFF_XP1    * xp
                         + COEFF_YM1    * ym
                         + COEFF_YP1    * yp
                         + COEFF_ZM1    * zm
                         + COEFF_ZP1    * zp;
        }

        __syncthreads();

        // Roll planes: inPrev ← inCurr, inCurr ← inNext
        // Use cooperative copying across threads
        for (int offset = 0; offset < (BLOCK_DIM + 2) * (BLOCK_DIM + 2);
             offset += blockDim.x * blockDim.y)
        {
            int flat = tx + ty * blockDim.x + offset;
            int lx = flat % (BLOCK_DIM + 2);
            int ly = flat / (BLOCK_DIM + 2);
            if (ly < BLOCK_DIM + 2) {
                inPrev_s[ly][lx] = inCurr_s[ly][lx];
                inCurr_s[ly][lx] = inNext_s[ly][lx];
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

    std::cout << "================================================\n";
    std::cout << "Chapter 8: Thread Coarsening Stencil (Fig 8.10)\n";
    std::cout << "================================================\n";
    std::cout << "Grid: " << N << " x " << N << " x " << N << "\n";
    std::cout << "Block (2D): " << BLOCK_DIM << " x " << BLOCK_DIM << "\n";
    std::cout << "Input plane (shared): " << (BLOCK_DIM+2) << " x " << (BLOCK_DIM+2) << "\n";
    std::cout << "Shared mem: 3 x " << ((BLOCK_DIM+2)*(BLOCK_DIM+2)*4) << " B = "
              << (3*(BLOCK_DIM+2)*(BLOCK_DIM+2)*sizeof(float)/1024) << " KB\n";
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

    // Grid: 2D block, 3D grid (z grid covers interior planes with stride TILE_DIM-2)
    dim3 blockDim(BLOCK_DIM, BLOCK_DIM);
    dim3 gridDim(
        (N + BLOCK_DIM - 1) / BLOCK_DIM,
        (N + BLOCK_DIM - 1) / BLOCK_DIM,
        ((N - 2) + (TILE_DIM - 2) - 1) / (TILE_DIM - 2)   // ceil((N-2)/(TILE_DIM-2))
    );

    std::cout << "Grid:  " << gridDim.x << " x " << gridDim.y << " x " << gridDim.z << "\n";
    std::cout << "Blocks: " << gridDim.x * gridDim.y * gridDim.z << "\n\n";

    // Warm-up
    coarsened_stencil_kernel<<<gridDim, blockDim>>>(d_in, d_out, N);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Timed run
    gpu_timer timer;
    timer.start();
    coarsened_stencil_kernel<<<gridDim, blockDim>>>(d_in, d_out, N);
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();

    float kernel_time = timer.elapsed_ms();

    // Copy result back
    CHECK_CUDA(cudaMemcpy(h_out, d_out, total * sizeof(float), cudaMemcpyDeviceToHost));

    // Metrics
    int interior = (N-2) * (N-2) * (N-2);
    float total_ops = interior * 13.0f;
    float gflops = total_ops / (kernel_time * 1e6f);

    float bytes_loaded = gridDim.x * gridDim.y * 3 * (BLOCK_DIM+2) * (BLOCK_DIM+2) * sizeof(float) * (float)gridDim.z;  // approximate
    float bytes_stored = interior * sizeof(float);
    // More accurate: each (x,y) block loads 3 planes per z step plus the initial 2
    // But for approximate BW we use total data touched
    float approx_load = gridDim.x * gridDim.y * (BLOCK_DIM+2) * (BLOCK_DIM+2) * sizeof(float) * (3 + 2 * ((N-2) - 2)) + interior * sizeof(float);
    float eff_bandwidth = ((approx_load + bytes_stored) / (kernel_time * 1e-3f)) / (1.0737e9f);

    // Validate
    bool valid = cpu_allclose(h_ref, h_out, total);

    std::cout << "Results:\n";
    std::cout << "  Kernel time:     " << std::fixed << std::setprecision(3) << kernel_time << " ms\n";
    std::cout << "  Interior cells:  " << interior << "\n";
    std::cout << "  GFLOPS:          " << std::fixed << std::setprecision(2) << gflops << "\n";
    std::cout << "  OP/B ratio:      " << std::fixed << std::setprecision(2)
              << (13.0f / 4.0f) * std::pow(1.0f - 2.0f/BLOCK_DIM, 3.0f) << " (theoretical)\n";
    std::cout << "  Validation:      " << (valid ? "PASSED" : "FAILED") << "\n\n";

    // Cleanup
    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_out));
    delete[] h_in;
    delete[] h_out;
    delete[] h_ref;

    return valid ? 0 : 1;
}
