/*
 * Chapter 8: Stencil
 * Shared Memory Tiling for Stencil Sweep — Fig 8.8
 *
 * Implements a 3D seven-point stencil sweep with shared memory tiling.
 * Each thread block loads an 8×8×8 input tile into shared memory,
 * then threads compute 6×6×6 output grid points.
 *
 * Key insight: input tiles are larger than output tiles because
 * halo cells (ghost cells) at boundaries are needed for stencil computation.
 * For order-1 stencil, 1 halo cell is needed on each side.
 *
 * Block size = 8×8×8 = 512 threads
 * Input tile = (TILE_DIM+2) × (TILE_DIM+2) × (TILE_DIM+2) = 8×8×8
 * Output tile = TILE_DIM × TILE_DIM × TILE_DIM = 6×6×6
 *
 * Limitations:
 *   - 58% of input tile elements are halos (poor reuse)
 *   - 1.37 OP/B arithmetic intensity at T=8
 *   - Poor coalescing: 8×8×8 block → warp accesses 4 different rows
 *
 * Compute: GTX 1050, sm_61 (device 1)
 */

#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstring>
#include "../common/cuda_utils.cuh"

/* Stencil coefficients (same as basic kernel) */
#define COEFF_CENTER  -6.0f
#define COEFF_XM1      1.0f
#define COEFF_XP1      1.0f
#define COEFF_YM1      1.0f
#define COEFF_YP1      1.0f
#define COEFF_ZM1      1.0f
#define COEFF_ZP1      1.0f

/* Tile configuration */
#define TILE_DIM   6   /* output tile dimension */
#define TILE_PAD   1   /* halo cells on each side (order 1) */
#define BLOCK_DIM  (TILE_DIM + 2 * TILE_PAD)  /* 8: input tile dimension = block size */

/*
 * Tiled 3D seven-point stencil kernel (Fig 8.8).
 *
 * Each block loads a BLOCK_DIM^3 input tile into shared memory.
 * Each thread loads exactly one input element.
 * Active threads (those within TILE_DIM in each dim) compute output values.
 *
 * Parameters:
 *   in   - 3D input grid (N x N x N, row-major)
 *   out  - 3D output grid (N x N x N, row-major)
 *   N    - grid dimension in each axis
 */
__global__ void tiled_stencil_kernel(
    const float* __restrict__ in,
    float* __restrict__ out,
    int N)
{
    // Shared memory input tile: BLOCK_DIM^3 (8×8×8)
    __shared__ float in_s[BLOCK_DIM][BLOCK_DIM][BLOCK_DIM];

    // Thread index within block
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;

    // Input tile coordinates (offset by -TILE_PAD for halo)
    // The input tile starts TILE_PAD cells before the output tile
    int i_in = blockIdx.x * TILE_DIM + tx - TILE_PAD;
    int j_in = blockIdx.y * TILE_DIM + ty - TILE_PAD;
    int k_in = blockIdx.z * TILE_DIM + tz - TILE_PAD;

    // Load input tile element into shared memory
    // Guard against ghost cells (out-of-bounds accesses at grid edges)
    if (i_in >= 0 && i_in < N &&
        j_in >= 0 && j_in < N &&
        k_in >= 0 && k_in < N)
    {
        in_s[tz][ty][tx] = in[k_in * N * N + j_in * N + i_in];
    }
    else
    {
        in_s[tz][ty][tx] = 0.0f;  // ghost cells = zero
    }

    __syncthreads();

    // Output tile coordinates (absolute grid positions)
    int i_out = blockIdx.x * TILE_DIM + tx;
    int j_out = blockIdx.y * TILE_DIM + ty;
    int k_out = blockIdx.z * TILE_DIM + tz;

    // Only compute if:
    // 1. Not a boundary cell (i,j,k >= 1 and < N-1)
    // 2. Within output tile (tx, ty, tz < TILE_DIM)
    if (i_out >= 1 && i_out < N-1 &&
        j_out >= 1 && j_out < N-1 &&
        k_out >= 1 && k_out < N-1 &&
        tx < TILE_DIM && ty < TILE_DIM && tz < TILE_DIM)
    {
        // Shared memory indices (offset by TILE_PAD for halo)
        // Since tx/ty/tz < TILE_DIM, the halo offset means:
        //   s[tz]   = s[TILE_PAD+tz-1] = s[tz]   (center)
        //   s[tz-1] = s[TILE_PAD+tz-1] = s[tz-1] (z-1 neighbor)
        //   s[tz+1] = s[TILE_PAD+tz+1] = s[tz+1] (z+1 neighbor)
        // Same for tx, ty

        float center = in_s[tz + TILE_PAD][ty + TILE_PAD][tx + TILE_PAD];
        float xm1    = in_s[tz + TILE_PAD][ty + TILE_PAD][tx + TILE_PAD - 1];
        float xp1    = in_s[tz + TILE_PAD][ty + TILE_PAD][tx + TILE_PAD + 1];
        float ym1    = in_s[tz + TILE_PAD][ty + TILE_PAD - 1][tx + TILE_PAD];
        float yp1    = in_s[tz + TILE_PAD][ty + TILE_PAD + 1][tx + TILE_PAD];
        float zm1    = in_s[tz + TILE_PAD - 1][ty + TILE_PAD][tx + TILE_PAD];
        float zp1    = in_s[tz + TILE_PAD + 1][ty + TILE_PAD][tx + TILE_PAD];

        int out_idx = k_out * N * N + j_out * N + i_out;
        out[out_idx] = COEFF_CENTER * center
                     + COEFF_XM1    * xm1
                     + COEFF_XP1    * xp1
                     + COEFF_YM1    * ym1
                     + COEFF_YP1    * yp1
                     + COEFF_ZM1    * zm1
                     + COEFF_ZP1    * zp1;
    }
}

/* -------------------------------------------------------------------------- */
/*  CPU reference implementation                                              */
/* -------------------------------------------------------------------------- */
__host__ void cpu_stencil_sweep(
    const float* in,
    float* out,
    int N)
{
    for (int k = 1; k < N-1; ++k) {
        for (int j = 1; j < N-1; ++j) {
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

    std::cout << "============================================\n";
    std::cout << "Chapter 8: Tiled Stencil Sweep (Fig 8.8)\n";
    std::cout << "============================================\n";
    std::cout << "Grid: " << N << " x " << N << " x " << N << "\n";
    std::cout << "Total elements: " << total << "\n";
    std::cout << "Block: " << BLOCK_DIM << " x " << BLOCK_DIM << " x " << BLOCK_DIM << "\n";
    std::cout << "Output tile: " << TILE_DIM << " x " << TILE_DIM << " x " << TILE_DIM << "\n";
    std::cout << "Device: GTX 1050 (sm_61)\n\n";

    // Host memory
    float* h_in  = new float[total];
    float* h_out = new float[total];
    float* h_ref = new float[total];

    // Initialize with 3D sine function
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

    // Grid configuration: one block per output tile
    dim3 blockDim(BLOCK_DIM, BLOCK_DIM, BLOCK_DIM);
    dim3 gridDim(
        (N + TILE_DIM - 1) / TILE_DIM,
        (N + TILE_DIM - 1) / TILE_DIM,
        (N + TILE_DIM - 1) / TILE_DIM
    );

    std::cout << "Grid:  " << gridDim.x << " x " << gridDim.y << " x " << gridDim.z << "\n";
    std::cout << "Blocks: " << gridDim.x * gridDim.y * gridDim.z << "\n\n";

    // Warm-up
    tiled_stencil_kernel<<<gridDim, blockDim>>>(d_in, d_out, N);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Timed run
    gpu_timer timer;
    timer.start();
    tiled_stencil_kernel<<<gridDim, blockDim>>>(d_in, d_out, N);
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();

    float kernel_time = timer.elapsed_ms();

    // Copy result back
    CHECK_CUDA(cudaMemcpy(h_out, d_out, total * sizeof(float), cudaMemcpyDeviceToHost));

    // Metrics
    int interior = (N-2) * (N-2) * (N-2);
    float total_ops = interior * 13.0f;
    float gflops = total_ops / (kernel_time * 1e6f);

    // Compute actual data touched from global memory
    int total_blocks = gridDim.x * gridDim.y * gridDim.z;
    float bytes_loaded = total_blocks * BLOCK_DIM * BLOCK_DIM * BLOCK_DIM * sizeof(float);
    float bytes_stored = interior * sizeof(float);
    float eff_bandwidth = ((bytes_loaded + bytes_stored) / (kernel_time * 1e-3f)) / (1024.0f * 1024.0f * 1024.0f);

    // Validate
    bool valid = cpu_allclose(h_ref, h_out, total);

    std::cout << "Results:\n";
    std::cout << "  Kernel time:      " << std::fixed << std::setprecision(3) << kernel_time << " ms\n";
    std::cout << "  Interior cells:   " << interior << "\n";
    std::cout << "  Tiles loaded:     " << total_blocks << " x " << (BLOCK_DIM*BLOCK_DIM*BLOCK_DIM) << " = "
              << (total_blocks * BLOCK_DIM * BLOCK_DIM * BLOCK_DIM) << "\n";
    std::cout << "  GFLOPS:           " << std::fixed << std::setprecision(2) << gflops << "\n";
    std::cout << "  Eff BW (load+st): " << std::fixed << std::setprecision(1) << eff_bandwidth << " GB/s\n";
    std::cout << "  OP/B ratio:       " << std::fixed << std::setprecision(2)
              << (13.0f / 4.0f) * std::pow(1.0f - 2.0f/BLOCK_DIM, 3.0f) << " (theoretical)\n";
    std::cout << "  Validation:       " << (valid ? "PASSED" : "FAILED") << "\n\n";

    // Cleanup
    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_out));
    delete[] h_in;
    delete[] h_out;
    delete[] h_ref;

    return valid ? 0 : 1;
}
