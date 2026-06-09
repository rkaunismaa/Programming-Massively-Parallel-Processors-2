/*
 * Chapter 8: Stencil
 * Basic Parallel Stencil Sweep — Fig 8.6
 *
 * Implements a 3D seven-point stencil sweep kernel.
 * Each thread computes one output grid point value by
 * loading 7 input grid points and applying coefficients.
 *
 * Grid dimensions: N x N x N (cube)
 * Stencil: 3D seven-point (center + x, y, z neighbors of order 1)
 *
 * Boundary cells hold boundary conditions and are NOT updated
 * (only interior cells are computed).
 *
 * Compute: GTX 1050, sm_61 (device 1)
 */

#include <iostream>
#include <iomanip>
#include <cmath>
#include <cfloat>
#include "../common/cuda_utils.cuh"

/*
 * Coefficients for 3D seven-point Laplacian stencil.
 * Finite difference approximation:
 *   ∇²f ≈ c0*f(i,j,k) + c1*f(i-1,j,k) + c2*f(i+1,j,k)
 *       + c3*f(i,j-1,k) + c4*f(i,j+1,k)
 *       + c5*f(i,j,k-1) + c6*f(i,j,k+1)
 * where c0 = -6, c1..c6 = 1 for a standard 7-point Laplacian.
 */
#define COEFF_CENTER  -6.0f
#define COEFF_XM1      1.0f
#define COEFF_XP1      1.0f
#define COEFF_YM1      1.0f
#define COEFF_YP1      1.0f
#define COEFF_ZM1      1.0f
#define COEFF_ZP1      1.0f

/*
 * Basic 3D seven-point stencil kernel (Fig 8.6).
 *
 * Each thread computes one output grid point.
 * Loads 7 input values from global memory.
 * Arithmetic intensity: 13 OPs / (7 * 4 bytes) = 0.46 OP/B
 *
 * Parameters:
 *   in   - 3D input grid (N x N x N, row-major)
 *   out  - 3D output grid (N x N x N, row-major)
 *   N    - grid dimension in each axis
 */
__global__ void basic_stencil_kernel(
    const float* __restrict__ in,
    float* __restrict__ out,
    int N)
{
    // Compute grid coordinates for this thread
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // x (column)
    int j = blockIdx.y * blockDim.y + threadIdx.y;  // y (row)
    int k = blockIdx.z * blockDim.z + threadIdx.z;  // z (depth)

    // Only interior cells are computed (boundary cells preserved)
    if (i >= 1 && i < N-1 &&
        j >= 1 && j < N-1 &&
        k >= 1 && k < N-1)
    {
        // Linear index: row-major order (z * N^2 + y * N + x)
        int idx = k * N * N + j * N + i;

        // 3D seven-point stencil: center + 6 neighbors
        float center = in[idx];
        float xm1    = in[idx - 1];        // i-1, j, k
        float xp1    = in[idx + 1];        // i+1, j, k
        float ym1    = in[idx - N];        // i, j-1, k
        float yp1    = in[idx + N];        // i, j+1, k
        float zm1    = in[idx - N*N];      // i, j, k-1
        float zp1    = in[idx + N*N];      // i, j, k+1

        // Apply stencil coefficients: 7 multiplies + 6 additions = 13 OPs
        out[idx] = COEFF_CENTER * center
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
void cpu_stencil_sweep(
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
    // Select GTX 1050
    cudaSetDevice(1);
    CHECK_CUDA(cudaGetLastError());

    // Grid dimensions (cube)
    const int N = 64;

    // Total elements
    const int total = N * N * N;

    std::cout << "========================================\n";
    std::cout << "Chapter 8: Basic Stencil Sweep (Fig 8.6)\n";
    std::cout << "========================================\n";
    std::cout << "Grid: " << N << " x " << N << " x " << N << "\n";
    std::cout << "Total elements: " << total << "\n";
    std::cout << "Device: GTX 1050 (sm_61)\n\n";

    // Allocate host memory
    float* h_in     = new float[total];
    float* h_out    = new float[total];
    float* h_ref    = new float[total];

    // Initialize input with a known function: f(i,j,k) = sin(i*0.1) * cos(j*0.1) * sin(k*0.1)
    for (int k = 0; k < N; ++k) {
        for (int j = 0; j < N; ++j) {
            for (int i = 0; i < N; ++i) {
                int idx = k * N * N + j * N + i;
                h_in[idx] = sinf(i * 0.1f) * cosf(j * 0.1f) * sinf(k * 0.1f);
            }
        }
    }

    // Compute CPU reference (copy input first so boundaries are preserved)
    memcpy(h_ref, h_in, total * sizeof(float));
    cpu_stencil_sweep(h_in, h_ref, N);

    // Allocate device memory
    float* d_in  = nullptr;
    float* d_out = nullptr;
    CHECK_CUDA(cudaMalloc(&d_in,  total * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_out, total * sizeof(float)));

    // Copy input to device
    CHECK_CUDA(cudaMemcpy(d_in, h_in, total * sizeof(float), cudaMemcpyHostToDevice));

    // Initialize output with boundary values (copy from input)
    CHECK_CUDA(cudaMemcpy(d_out, h_in, total * sizeof(float), cudaMemcpyHostToDevice));

    // Launch configuration
    // Each interior thread computes one output point
    dim3 blockDim(8, 8, 8);   // 512 threads per block
    dim3 gridDim(
        (N + blockDim.x - 1) / blockDim.x,
        (N + blockDim.y - 1) / blockDim.y,
        (N + blockDim.z - 1) / blockDim.z
    );

    std::cout << "Block: " << blockDim.x << " x " << blockDim.y << " x " << blockDim.z << "\n";
    std::cout << "Grid:  " << gridDim.x << " x " << gridDim.y << " x " << gridDim.z << "\n";
    std::cout << "Threads: " << gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z << "\n\n";

    // Warm-up run
    basic_stencil_kernel<<<gridDim, blockDim>>>(d_in, d_out, N);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Timed run
    gpu_timer timer;
    timer.start();
    basic_stencil_kernel<<<gridDim, blockDim>>>(d_in, d_out, N);
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();

    float kernel_time = timer.elapsed_ms();

    // Copy result back
    CHECK_CUDA(cudaMemcpy(h_out, d_out, total * sizeof(float), cudaMemcpyDeviceToHost));

    // Compute effective bandwidth (only interior cells computed)
    int interior = (N-2) * (N-2) * (N-2);
    float bytes_read  = interior * 7 * sizeof(float);   // 7 loads per interior cell
    float bytes_write = interior * 1 * sizeof(float);   // 1 store per interior cell
    float total_bytes = bytes_read + bytes_write;
    float eff_bandwidth = (total_bytes / (kernel_time * 1e-3f)) / (1024.0f * 1024.0f * 1024.0f);

    // Count actual output cells (interior)
    int output_cells = 0;
    for (int i = 0; i < total; ++i) {
        if (h_out[i] != h_in[i]) output_cells++;  // boundary unchanged
    }

    // Validate (use cuda_utils.cuh cpu_allclose)
    bool valid = cpu_allclose(h_ref, h_out, total);

    std::cout << "Results:\n";
    std::cout << "  Kernel time:  " << std::fixed << std::setprecision(3) << kernel_time << " ms\n";
    std::cout << "  Interior:     " << interior << " cells\n";
    std::cout << "  Written:      " << output_cells << " cells\n";
    std::cout << "  Eff BW:       " << std::fixed << std::setprecision(1) << eff_bandwidth << " GB/s\n";
    std::cout << "  Validation:   " << (valid ? "PASSED" : "FAILED") << "\n";

    // Compute arithmetic intensity
    float ops_per_interior = 13.0f;  // 7 mul + 6 add
    float total_ops = interior * ops_per_interior;
    float gflops = total_ops / (kernel_time * 1e6f);
    std::cout << "  GFLOPS:       " << std::fixed << std::setprecision(2) << gflops << "\n";
    std::cout << "  OP/B ratio:   " << std::fixed << std::setprecision(2)
              << ops_per_interior / (7.0f * sizeof(float)) << "\n\n";

    // Cleanup
    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_out));
    delete[] h_in;
    delete[] h_out;
    delete[] h_ref;

    return valid ? 0 : 1;
}
