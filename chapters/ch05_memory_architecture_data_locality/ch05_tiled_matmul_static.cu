/*
 * Chapter 5: Memory Architecture and Data Locality
 * Tiled Matrix Multiplication - Static Shared Memory
 *
 * This kernel uses shared memory tiles to reduce global memory traffic.
 * Each thread block computes a TILE_WIDTH x TILE_WIDTH tile of the output matrix P.
 * The tiles of M and N are loaded into shared memory and reused across multiple iterations.
 *
 * Key concept: Tiling reduces global memory accesses from 2*N per output element
 * to 2*TILE_WIDTH per output element (where TILE_WIDTH << N).
 *
 * Compute: GTX 1050, sm_61
 */

#include <iostream>
#include <iomanip>
#include <cmath>
#include "../common/cuda_utils.cuh"

#define TILE_WIDTH 16

/*
 * Tiled matrix multiplication kernel.
 *
 * Each block computes one TILE_WIDTH x TILE_WIDTH tile of P.
 * Within each block, threads cooperate to:
 * 1. Load a tile of M and a tile of N into shared memory
 * 2. Compute partial dot products using the shared tiles
 * 3. Repeat for all tiles along the reduction dimension
 *
 * Shared memory layout:
 *   Mds[TILE_WIDTH][TILE_WIDTH] - tile from matrix M
 *   Nds[TILE_WIDTH][TILE_WIDTH] - tile from matrix N
 *
 * For an N x N matrix with TILE_WIDTH=16 and block size 16x16:
 * - Each thread computes one element of P
 * - Each thread accesses TILE_WIDTH elements per tile, total N/TILE_WIDTH tiles
 * - Global memory accesses per thread: 2 * TILE_WIDTH * (N/TILE_WIDTH) = 2*N (same total)
 *   BUT the data is shared within the block, so effective traffic is reduced by TILE_WIDTH
 */
__global__ void tiled_matmul_kernel(
    const float* M,
    const float* N,
    float* P,
    int width)
{
    // Shared memory tiles - allocated per block
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    // Thread indices within the block
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Row and column of the P element that this thread will compute
    int row = blockIdx.y * TILE_WIDTH + ty;
    int col = blockIdx.x * TILE_WIDTH + tx;

    // Register accumulator for the dot product
    float Pvalue = 0.0f;

    // Loop over all tiles along the reduction dimension
    int num_tiles = width / TILE_WIDTH;
    for (int t = 0; t < num_tiles; ++t)
    {
        // Collaborative loading: each thread loads one element into each shared tile
        // M tile: row is fixed for this block, column advances by tiles
        Mds[ty][tx] = M[row * width + t * TILE_WIDTH + tx];

        // N tile: column is fixed for this block, row advances by tiles
        Nds[ty][tx] = N[(t * TILE_WIDTH + ty) * width + col];

        // Wait for all threads in the block to finish loading
        __syncthreads();

        // Compute partial dot product using the shared tiles
        for (int k = 0; k < TILE_WIDTH; ++k)
        {
            Pvalue += Mds[ty][k] * Nds[k][tx];
        }

        // Wait before loading the next tile
        __syncthreads();
    }

    // Write the result to global memory
    P[row * width + col] = Pvalue;
}

int main()
{
    const int width = 4096;
    const int bytes = width * width * sizeof(float);

    std::cout << "Chapter 5: Tiled Matrix Multiplication (Static Shared Memory)" << std::endl;
    std::cout << "Matrix size: " << width << " x " << width << std::endl;
    std::cout << "Tile size: " << TILE_WIDTH << " x " << TILE_WIDTH << std::endl;
    std::cout << std::endl;

    // Target GTX 1050 (CUDA device 1)
    CHECK_CUDA(cudaSetDevice(1));

    // Allocate host memory
    float* h_M = new float[width * width];
    float* h_N = new float[width * width];
    float* h_P = new float[width * width];

    // Initialize matrices
    for (int i = 0; i < width * width; ++i)
    {
        h_M[i] = static_cast<float>(rand() % 100) / 100.0f;
        h_N[i] = static_cast<float>(rand() % 100) / 100.0f;
    }

    // Allocate device memory
    float *d_M, *d_N, *d_P;
    CHECK_CUDA(cudaMalloc((void**)&d_M, bytes));
    CHECK_CUDA(cudaMalloc((void**)&d_N, bytes));
    CHECK_CUDA(cudaMalloc((void**)&d_P, bytes));

    // Copy data to device
    CHECK_CUDA(cudaMemcpy(d_M, h_M, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_N, h_N, bytes, cudaMemcpyHostToDevice));

    // Configure kernel launch
    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid(width / TILE_WIDTH, width / TILE_WIDTH);

    std::cout << "Grid: " << grid.x << " x " << grid.y << " blocks" << std::endl;
    std::cout << "Block: " << block.x << " x " << block.y << " threads" << std::endl;
    std::cout << "Total threads: " << (grid.x * grid.y * block.x * block.y) << std::endl;
    std::cout << std::endl;

    // Warmup run
    tiled_matmul_kernel<<<grid, block>>>(d_M, d_N, d_P, width);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Timed run
    gpu_timer timer;
    timer.start();
    tiled_matmul_kernel<<<grid, block>>>(d_M, d_N, d_P, width);
    timer.stop();
    float elapsed_ms = timer.elapsed_ms();

    std::cout << "Kernel time: " << std::fixed << std::setprecision(2) << elapsed_ms << " ms" << std::endl;

    // Calculate GFLOPS
    // Matrix multiplication: 2 * N^3 FLOPs (multiply + add for each of N^3 operations)
    double flops = 2.0 * width * width * width;
    double gflops = (flops / (elapsed_ms / 1000.0)) / 1e9;
    std::cout << "Performance: " << std::fixed << std::setprecision(2) << gflops << " GFLOPS" << std::endl;

    // Copy result back and verify
    CHECK_CUDA(cudaMemcpy(h_P, d_P, bytes, cudaMemcpyDeviceToHost));

    // Simple verification: check a few elements
    bool correct = true;
    int check_count = 0;
    for (int i = 0; i < std::min(width, 64); ++i)
    {
        for (int j = 0; j < std::min(width, 64); ++j)
        {
            float expected = 0.0f;
            for (int k = 0; k < width; ++k)
            {
                expected += h_M[i * width + k] * h_N[k * width + j];
            }
            if (std::abs(h_P[i * width + j] - expected) > 1.0f)
            {
                correct = false;
                std::cout << "Mismatch at (" << i << "," << j << "): "
                          << h_P[i * width + j] << " vs " << expected << std::endl;
                break;
            }
            check_count++;
        }
    }

    std::cout << "Verification: " << (correct ? "PASSED" : "FAILED") << " (" << check_count << " elements checked)" << std::endl;

    // Cleanup
    CHECK_CUDA(cudaFree(d_M));
    CHECK_CUDA(cudaFree(d_N));
    CHECK_CUDA(cudaFree(d_P));
    delete[] h_M;
    delete[] h_N;
    delete[] h_P;

    return 0;
}
