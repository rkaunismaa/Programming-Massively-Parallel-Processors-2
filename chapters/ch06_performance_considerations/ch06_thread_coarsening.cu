/*
 * =============================================================================
 *  Chapter 6: Performance Considerations — Thread Coarsening
 * =============================================================================
 *  Book:      Programming Massively Parallel Processors (Kirk, Hwu & El Hajj,
 *             4th Edition, Morgan Kaufmann 2023)
 *  Figure:    6.13 — Thread-coarsened tiled matrix multiplication
 *  Purpose:   Demonstrate thread coarsening as a performance optimization
 *             technique that increases parallelism by having each thread
 *             compute multiple output elements.
 *  Hardware:  GTX 1050, sm_61 (Pascal) — all code targets this GPU
 * =============================================================================
 *
 *  THREAD COARSENING CONCEPT
 *  ---------------------------------------------------------------------------
 *  In the standard tiled matmul (Ch. 5), each thread computes ONE element of
 *  the output matrix P.  This leaves threads under-utilised during the global
 *  memory load phase because only a fraction of registers are active.
 *
 *  Thread coarsening assigns each thread multiple output columns (COARSE_FACTOR
 *  of them).  Concretely:
 *    - Each thread still loads the same M tile (one row), but now accumulates
 *      COARSE_FACTOR partial dot products in registers (Pvalue[COARSE_FACTOR]).
 *    - The inner coarsening loop iterates over the assigned columns, loading
 *      the corresponding N tile column and accumulating into the matching
 *      Pvalue entry.
 *    - The grid is shrunk in the x-dimension by COARSE_FACTOR because each
 *      block now covers COARSE_FACTOR * TILE_WIDTH columns instead of one tile.
 *
 *  Benefits:
 *    1. Higher arithmetic intensity — more FLOPs per global memory access.
 *    2. Better register utilisation during the compute phase.
 *    3. Fewer blocks launched (smaller grid), reducing launch overhead.
 *
 *  Trade-off:
 *    - More registers per thread (Pvalue array + loop counters).
 *    - May reduce occupancy if register pressure becomes too high.
 * =============================================================================
 */

#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstdlib>
#include "../common/cuda_utils.cuh"

#define TILE_WIDTH    32
#define COARSE_FACTOR 4

/*
 * Thread-coarsened tiled matrix multiplication kernel (Figure 6.13).
 *
 * Each thread computes COARSE_FACTOR elements of P along the column dimension.
 * Grid x-dimension is divided by COARSE_FACTOR compared to the standard tiled
 * version because each block covers (TILE_WIDTH * COARSE_FACTOR) columns.
 *
 * Parameters:
 *   M    - input matrix M (width x width), row-major
 *   N    - input matrix N (width x width), row-major
 *   P    - output matrix P = M * N (width x width), row-major
 *   width - dimension of the square matrices
 *
 * Requirements:
 *   width must be divisible by TILE_WIDTH (for tiling)
 *   width must be divisible by TILE_WIDTH * COARSE_FACTOR (for coarsening)
 */
__global__ void matrixMulKernel(float* M, float* N, float* P, int width)
{
    // Shared memory tiles — one per block
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    // Block and thread indices
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Identify the row and column of the P element to work on
    // Row: same as standard tiled — each block handles TILE_WIDTH rows
    int row = by * TILE_WIDTH + ty;
    // Column start: each block handles (TILE_WIDTH * COARSE_FACTOR) columns
    // tx selects which of the TILE_WIDTH starting positions within the block's range
    int colStart = bx * TILE_WIDTH * COARSE_FACTOR + tx;

    // Initialize Pvalue for all COARSE_FACTOR output elements
    float Pvalue[COARSE_FACTOR];
    for (int c = 0; c < COARSE_FACTOR; ++c) {
        Pvalue[c] = 0.0f;
    }

    // Loop over the M and N tiles required to compute the P elements
    // Number of tiles along the reduction dimension = width / TILE_WIDTH
    for (int ph = 0; ph < width / TILE_WIDTH; ++ph) {

        // Collaborative loading of M tile into shared memory
        // Each thread loads one element: row is fixed, column advances by tiles
        Mds[ty][tx] = M[row * width + ph * TILE_WIDTH + tx];

        // Coarsening loop: iterate over the COARSE_FACTOR columns
        for (int c = 0; c < COARSE_FACTOR; ++c) {

            // Column for this coarsening iteration
            int col = colStart + c * TILE_WIDTH;

            // Collaborative loading of N tile into shared memory
            // Column varies with c, row advances by tiles
            Nds[ty][tx] = N[(ph * TILE_WIDTH + ty) * width + col];

            // Synchronize: ensure M and N tiles are fully loaded
            __syncthreads();

            // Compute partial dot product for column c
            for (int k = 0; k < TILE_WIDTH; ++k) {
                Pvalue[c] += Mds[ty][k] * Nds[k][tx];
            }

            // Synchronize before loading the next tile pair
            __syncthreads();
        }
    }

    // Write all COARSE_FACTOR results to global memory
    for (int c = 0; c < COARSE_FACTOR; ++c) {
        int col = colStart + c * TILE_WIDTH;
        P[row * width + col] = Pvalue[c];
    }
}

/*
 * CPU reference: straightforward triple-loop matrix multiplication.
 * Used for validation against the GPU result.
 */
void cpu_matmul(const float* M, const float* N, float* P, int width)
{
    for (int i = 0; i < width; ++i) {
        for (int j = 0; j < width; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < width; ++k) {
                sum += M[i * width + k] * N[k * width + j];
            }
            P[i * width + j] = sum;
        }
    }
}

int main()
{
    // Matrix dimension — must be divisible by TILE_WIDTH * COARSE_FACTOR
    // 2048 / (32 * 4) = 2048 / 128 = 16 blocks in x-dimension
    const int width = 2048;
    const int bytes = width * width * sizeof(float);

    std::cout << "============================================================" << std::endl;
    std::cout << "Chapter 6: Thread Coarsening — Tiled Matrix Multiplication" << std::endl;
    std::cout << "============================================================" << std::endl;
    std::cout << "Matrix size     : " << width << " x " << width << std::endl;
    std::cout << "Tile size       : " << TILE_WIDTH << " x " << TILE_WIDTH << std::endl;
    std::cout << "Coarse factor   : " << COARSE_FACTOR << std::endl;
    std::cout << std::endl;

    // Target GTX 1050 (CUDA device 1)
    CHECK_CUDA(cudaSetDevice(1));

    // Print device information
    print_device_info(1);

    // Allocate host memory
    float* h_M = new float[width * width];
    float* h_N = new float[width * width];
    float* h_P_gpu = new float[width * width];
    float* h_P_cpu = new float[width * width];

    // Initialize matrices with simple deterministic values
    // Using small values to keep floating-point error manageable
    srand(42);
    for (int i = 0; i < width * width; ++i) {
        h_M[i] = static_cast<float>(rand() % 100) / 100.0f;
        h_N[i] = static_cast<float>(rand() % 100) / 100.0f;
    }

    // Allocate device memory
    float *d_M, *d_N, *d_P;
    CHECK_CUDA(cudaMalloc((void**)&d_M, bytes));
    CHECK_CUDA(cudaMalloc((void**)&d_N, bytes));
    CHECK_CUDA(cudaMalloc((void**)&d_P, bytes));

    // Copy input data to device
    CHECK_CUDA(cudaMemcpy(d_M, h_M, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_N, h_N, bytes, cudaMemcpyHostToDevice));

    // Configure kernel launch parameters
    // Grid x: width / (TILE_WIDTH * COARSE_FACTOR) = 2048 / 128 = 16
    // Grid y: width / TILE_WIDTH = 2048 / 32 = 64
    // Block: TILE_WIDTH x TILE_WIDTH = 32 x 32 = 1024 threads
    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid(width / (TILE_WIDTH * COARSE_FACTOR), width / TILE_WIDTH);

    std::cout << "Grid  : " << grid.x << " x " << grid.y << " blocks" << std::endl;
    std::cout << "Block : " << block.x << " x " << block.y << " threads" << std::endl;
    std::cout << "Threads per block : " << (block.x * block.y) << std::endl;
    std::cout << "Total blocks      : " << (grid.x * grid.y) << std::endl;
    std::cout << std::endl;

    // Warmup run — ensures kernel is JIT-compiled and caches are populated
    matrixMulKernel<<<grid, block>>>(d_M, d_N, d_P, width);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Timed run
    gpu_timer timer;
    timer.start();
    matrixMulKernel<<<grid, block>>>(d_M, d_N, d_P, width);
    timer.stop();
    float elapsed_ms = timer.elapsed_ms();

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_P_gpu, d_P, bytes, cudaMemcpyDeviceToHost));

    // Timing and performance output
    std::cout << "------------------------------------------------------------" << std::endl;
    std::cout << "Performance Results:" << std::endl;
    std::cout << "  Kernel time   : " << std::fixed << std::setprecision(2)
              << elapsed_ms << " ms" << std::endl;

    // GFLOPS = 2 * N^3 FLOPs / (time in seconds) / 1e9
    // The factor of 2 accounts for multiply + add per inner-product element
    double flops = 2.0 * width * width * width;
    double gflops = (flops / (elapsed_ms * 1e6));
    std::cout << "  GFLOPS        : " << std::fixed << std::setprecision(2)
              << gflops << " GFLOPS" << std::endl;
    std::cout << "  Total FLOPs   : " << std::fixed << std::setprecision(0)
              << flops << std::endl;
    std::cout << "------------------------------------------------------------" << std::endl;

    // Validation: compute CPU reference and compare
    std::cout << std::endl;
    std::cout << "Validation (CPU reference)..." << std::endl;
    cpu_matmul(h_M, h_N, h_P_cpu, width);

    bool passed = cpu_allclose(h_P_cpu, h_P_gpu, width * width, 1.0f);
    std::cout << "  Result: " << (passed ? "PASSED" : "FAILED") << std::endl;
    std::cout << std::endl;

    // Cleanup
    CHECK_CUDA(cudaFree(d_M));
    CHECK_CUDA(cudaFree(d_N));
    CHECK_CUDA(cudaFree(d_P));
    delete[] h_M;
    delete[] h_N;
    delete[] h_P_gpu;
    delete[] h_P_cpu;

    return 0;
}
