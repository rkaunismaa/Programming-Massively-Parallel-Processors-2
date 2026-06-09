/*
 * Chapter 5: Memory Architecture and Data Locality
 * Tiled Matrix Multiplication - Dynamic Shared Memory
 *
 * Same tiling approach as the static version, but uses dynamic shared memory
 * so the tile size can be specified at runtime. This demonstrates how shared
 * memory usage impacts occupancy - larger tiles mean more shared memory per
 * block, which reduces the number of blocks that can reside on each SM.
 *
 * Dynamic shared memory is allocated via the 3rd argument to the kernel launch:
 *   kernel<<<grid, block, shared_mem_bytes>>>(args...);
 *
 * Access dynamic shared memory via a pointer:
 *   extern __shared__ float shared_mem[];
 *
 * Compute: GTX 1050, sm_61
 */

#include <iostream>
#include <iomanip>
#include <cmath>
#include "../common/cuda_utils.cuh"

/*
 * Tiled matrix multiplication with dynamic shared memory.
 *
 * The tile width is passed as a parameter, and shared memory is allocated
 * dynamically at kernel launch time. This allows experimenting with different
 * tile sizes to find the optimal configuration for a given hardware.
 */
__global__ void tiled_matmul_dynamic_kernel(
    const float* M,
    const float* N,
    float* P,
    int width,
    int tile_width)
{
    // Dynamic shared memory - size determined at launch
    extern __shared__ float shared_mem[];

    // Split shared memory into two tiles
    float* Mds = shared_mem;              // First half: M tile
    float* Nds = shared_mem + tile_width * tile_width; // Second half: N tile

    // Thread indices within the block
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Row and column of the P element that this thread will compute
    int row = blockIdx.y * tile_width + ty;
    int col = blockIdx.x * tile_width + tx;

    // Register accumulator for the dot product
    float Pvalue = 0.0f;

    // Loop over all tiles along the reduction dimension
    int num_tiles = width / tile_width;
    for (int t = 0; t < num_tiles; ++t)
    {
        // Collaborative loading into dynamic shared memory
        Mds[ty * tile_width + tx] = M[row * width + t * tile_width + tx];
        Nds[ty * tile_width + tx] = N[(t * tile_width + ty) * width + col];

        // Wait for all threads in the block to finish loading
        __syncthreads();

        // Compute partial dot product using the shared tiles
        for (int k = 0; k < tile_width; ++k)
        {
            Pvalue += Mds[ty * tile_width + k] * Nds[k * tile_width + tx];
        }

        // Wait before loading the next tile
        __syncthreads();
    }

    // Write the result to global memory
    P[row * width + col] = Pvalue;
}

int main(int argc, char** argv)
{
    const int width = 4096;
    int tile_width = 16; // Default, can be overridden

    if (argc > 1)
    {
        tile_width = atoi(argv[1]);
    }

    const int bytes = width * width * sizeof(float);

    std::cout << "Chapter 5: Tiled Matrix Multiplication (Dynamic Shared Memory)" << std::endl;
    std::cout << "Matrix size: " << width << " x " << width << std::endl;
    std::cout << "Tile size: " << tile_width << " x " << tile_width << std::endl;
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
    dim3 block(tile_width, tile_width);
    dim3 grid(width / tile_width, width / tile_width);

    // Calculate dynamic shared memory size: 2 tiles per block
    int shared_mem_bytes = 2 * tile_width * tile_width * sizeof(float);

    std::cout << "Grid: " << grid.x << " x " << grid.y << " blocks" << std::endl;
    std::cout << "Block: " << block.x << " x " << block.y << " threads" << std::endl;
    std::cout << "Dynamic shared memory per block: " << shared_mem_bytes << " bytes" << std::endl;
    std::cout << std::endl;

    // Warmup run
    tiled_matmul_dynamic_kernel<<<grid, block, shared_mem_bytes>>>(d_M, d_N, d_P, width, tile_width);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Timed run
    gpu_timer timer;
    timer.start();
    tiled_matmul_dynamic_kernel<<<grid, block, shared_mem_bytes>>>(d_M, d_N, d_P, width, tile_width);
    timer.stop();
    float elapsed_ms = timer.elapsed_ms();

    std::cout << "Kernel time: " << std::fixed << std::setprecision(2) << elapsed_ms << " ms" << std::endl;

    // Calculate GFLOPS
    double flops = 2.0 * width * width * width;
    double gflops = (flops / (elapsed_ms / 1000.0)) / 1e9;
    std::cout << "Performance: " << std::fixed << std::setprecision(2) << gflops << " GFLOPS" << std::endl;

    // Copy result back and verify
    CHECK_CUDA(cudaMemcpy(h_P, d_P, bytes, cudaMemcpyDeviceToHost));

    // Simple verification
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
