/*
 * Chapter 5: Memory Architecture and Data Locality
 * Tiled Matrix Multiplication - With Boundary Checks
 *
 * Handles matrices where dimensions are not evenly divisible by TILE_WIDTH.
 * Boundary checks ensure threads don't access out-of-bounds memory.
 *
 * Key insight: When matrix dimensions aren't multiples of TILE_WIDTH,
 * some threads in the last blocks will have row/col indices beyond the
 * matrix boundaries. These threads need to check bounds before loading
 * and storing data.
 *
 * Compute: GTX 1050, sm_61
 */

#include <iostream>
#include <iomanip>
#include <cmath>
#include "../common/cuda_utils.cuh"

#define TILE_WIDTH 16

/*
 * Tiled matrix multiplication with boundary checks.
 *
 * Handles rectangular matrices and matrices whose dimensions are not
 * multiples of TILE_WIDTH. Each thread checks whether its row/column
 * index is within bounds before accessing global memory.
 *
 * Note: The boundary check adds a branch that can cause warp divergence
 * in the last few blocks, but this affects only a small fraction of the
 * total computation for large matrices.
 */
__global__ void tiled_matmul_boundary_kernel(
    const float* M,
    const float* N,
    float* P,
    int M_width,
    int N_width)
{
    // Shared memory tiles
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
    // M_width == N_width (square matrices M and N share the inner dimension)
    int num_tiles = (M_width + TILE_WIDTH - 1) / TILE_WIDTH; // Ceiling division
    for (int t = 0; t < num_tiles; ++t)
    {
        // Collaborative loading with boundary checks
        // M tile: check if column index is within bounds
        int m_col = t * TILE_WIDTH + tx;
        if (row < M_width && m_col < M_width)
        {
            Mds[ty][tx] = M[row * M_width + m_col];
        }
        else
        {
            Mds[ty][tx] = 0.0f;
        }

        // N tile: check if row index is within bounds
        int n_row = t * TILE_WIDTH + ty;
        if (n_row < M_width && col < N_width)
        {
            Nds[ty][tx] = N[n_row * N_width + col];
        }
        else
        {
            Nds[ty][tx] = 0.0f;
        }

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

    // Write the result to global memory with boundary check
    if (row < M_width && col < N_width)
    {
        P[row * N_width + col] = Pvalue;
    }
}

int main(int argc, char** argv)
{
    int M_rows = 4097;  // Not divisible by TILE_WIDTH
    int M_width = 4097; // Inner dimension (shared with N)
    int N_width = 4097; // Not divisible by TILE_WIDTH

    if (argc > 1) M_rows = atoi(argv[1]);
    if (argc > 2) M_width = atoi(argv[2]);
    if (argc > 3) N_width = atoi(argv[3]);

    const int bytes_M = M_rows * M_width * sizeof(float);
    const int bytes_N = M_width * N_width * sizeof(float);
    const int bytes_P = M_rows * N_width * sizeof(float);

    std::cout << "Chapter 5: Tiled Matrix Multiplication (Boundary Checks)" << std::endl;
    std::cout << "Matrix M: " << M_rows << " x " << M_width << std::endl;
    std::cout << "Matrix N: " << M_width << " x " << N_width << std::endl;
    std::cout << "Result P: " << M_rows << " x " << N_width << std::endl;
    std::cout << "Tile size: " << TILE_WIDTH << " x " << TILE_WIDTH << std::endl;
    std::cout << std::endl;

    // Target GTX 1050 (CUDA device 1)
    CHECK_CUDA(cudaSetDevice(1));

    // Allocate host memory
    float* h_M = new float[M_rows * M_width];
    float* h_N = new float[M_width * N_width];
    float* h_P = new float[M_rows * N_width];

    // Initialize matrices
    for (int i = 0; i < M_rows * M_width; ++i)
    {
        h_M[i] = static_cast<float>(rand() % 100) / 100.0f;
    }
    for (int i = 0; i < M_width * N_width; ++i)
    {
        h_N[i] = static_cast<float>(rand() % 100) / 100.0f;
    }

    // Allocate device memory
    float *d_M, *d_N, *d_P;
    CHECK_CUDA(cudaMalloc((void**)&d_M, bytes_M));
    CHECK_CUDA(cudaMalloc((void**)&d_N, bytes_N));
    CHECK_CUDA(cudaMalloc((void**)&d_P, bytes_P));

    // Copy data to device
    CHECK_CUDA(cudaMemcpy(d_M, h_M, bytes_M, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_N, h_N, bytes_N, cudaMemcpyHostToDevice));

    // Configure kernel launch
    // Grid size: ceiling division to cover all elements
    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid((N_width + TILE_WIDTH - 1) / TILE_WIDTH,
              (M_rows + TILE_WIDTH - 1) / TILE_WIDTH);

    std::cout << "Grid: " << grid.x << " x " << grid.y << " blocks" << std::endl;
    std::cout << "Block: " << block.x << " x " << block.y << " threads" << std::endl;
    std::cout << std::endl;

    // Warmup run
    tiled_matmul_boundary_kernel<<<grid, block>>>(d_M, d_N, d_P, M_width, N_width);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Timed run
    gpu_timer timer;
    timer.start();
    tiled_matmul_boundary_kernel<<<grid, block>>>(d_M, d_N, d_P, M_width, N_width);
    timer.stop();
    float elapsed_ms = timer.elapsed_ms();

    std::cout << "Kernel time: " << std::fixed << std::setprecision(2) << elapsed_ms << " ms" << std::endl;

    // Calculate GFLOPS
    double flops = 2.0 * M_rows * M_width * N_width;
    double gflops = (flops / (elapsed_ms / 1000.0)) / 1e9;
    std::cout << "Performance: " << std::fixed << std::setprecision(2) << gflops << " GFLOPS" << std::endl;

    // Copy result back and verify
    CHECK_CUDA(cudaMemcpy(h_P, d_P, bytes_P, cudaMemcpyDeviceToHost));

    // Verification
    bool correct = true;
    int check_count = 0;
    int max_check_rows = std::min(M_rows, 64);
    int max_check_cols = std::min(N_width, 64);
    for (int i = 0; i < max_check_rows; ++i)
    {
        for (int j = 0; j < max_check_cols; ++j)
        {
            float expected = 0.0f;
            for (int k = 0; k < M_width; ++k)
            {
                expected += h_M[i * M_width + k] * h_N[k * N_width + j];
            }
            if (std::abs(h_P[i * N_width + j] - expected) > 1.0f)
            {
                correct = false;
                std::cout << "Mismatch at (" << i << "," << j << "): "
                          << h_P[i * N_width + j] << " vs " << expected << std::endl;
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
