/*
 * =============================================================================
 *  ch16_conv_forward.cu — Direct Convolutional Layer Forward Kernel
 * =============================================================================
 *  Book:      Programming Massively Parallel Processors (Kirk, Hwu & El Hajj,
 *             4th Edition, Morgan Kaufmann 2023)
 *  Chapter:   16 — Deep Learning
 *  Section:   16.3 — Convolutional Layer: A CUDA Inference Kernel (Fig 16.15)
 *
 *  Implements the forward pass of a 2D convolutional layer with multiple
 *  input/output channels and a minibatch of samples.
 *
 *  Data layout (row-major, linearized):
 *    X[N][C][H][W]           — input feature maps
 *    W_weights[M][C][K][K]   — filter banks (K×K kernel per channel pair)
 *    Y[N][M][H_out][W_out]   — output feature maps  (H_out = H-K+1, etc.)
 *
 *  Thread organization:
 *    2D thread blocks: TILE_WIDTH × TILE_WIDTH threads compute one tile
 *    3D grid: (M, T, N) where T = H_grid * W_grid is number of tiles per map
 *
 *  Hardware: GTX 1050, sm_61 (Pascal)
 * =============================================================================
 */

#include "../common/cuda_utils.cuh"
#include <cmath>
#include <cstring>

#define TILE_WIDTH 16

/* -------------------------------------------------------------------------- */
/*  GPU Kernel: ConvLayerForward (Fig 16.15)                                  */
/* -------------------------------------------------------------------------- */

__global__ void conv_forward_kernel(
    int C, int H, int W, int K, int W_grid,
    const float* X, const float* W_weights, float* Y)
{
    // Output dimensions
    int H_out = H - K + 1;
    int W_out = W - K + 1;

    // Identify which output pixel this thread computes
    int m = blockIdx.x;                                       // output feature map
    int h = (blockIdx.y / W_grid) * TILE_WIDTH + threadIdx.y; // vertical pixel
    int w = (blockIdx.y % W_grid) * TILE_WIDTH + threadIdx.x; // horizontal pixel
    int n = blockIdx.z;                                       // sample in minibatch

    if (h >= H_out || w >= W_out) return;

    float acc = 0.0f;

    // Sum over all input channels
    for (int c = 0; c < C; c++) {
        // Convolve K×K filter over input patch
        for (int p = 0; p < K; p++) {
            for (int q = 0; q < K; q++) {
                // X[n, c, h+p, w+q]
                int x_idx = ((n * C + c) * H + (h + p)) * W + (w + q);
                // W[m, c, p, q]
                int w_idx = ((m * C + c) * K + p) * K + q;
                acc += X[x_idx] * W_weights[w_idx];
            }
        }
    }

    // Y[n, m, h, w]
    int y_idx = ((n * gridDim.x + m) * H_out + h) * W_out + w;
    Y[y_idx] = acc;
}

/* -------------------------------------------------------------------------- */
/*  CPU Reference: sequential convolution for validation                      */
/* -------------------------------------------------------------------------- */

void cpu_conv_forward(
    int N, int C, int H, int W, int K, int M,
    const float* X, const float* W_weights, float* Y)
{
    int H_out = H - K + 1;
    int W_out = W - K + 1;

    for (int n = 0; n < N; n++) {
        for (int m = 0; m < M; m++) {
            for (int h = 0; h < H_out; h++) {
                for (int w = 0; w < W_out; w++) {
                    float acc = 0.0f;
                    for (int c = 0; c < C; c++) {
                        for (int p = 0; p < K; p++) {
                            for (int q = 0; q < K; q++) {
                                int x_idx = ((n * C + c) * H + (h + p)) * W + (w + q);
                                int w_idx = ((m * C + c) * K + p) * K + q;
                                acc += X[x_idx] * W_weights[w_idx];
                            }
                        }
                    }
                    int y_idx = ((n * M + m) * H_out + h) * W_out + w;
                    Y[y_idx] = acc;
                }
            }
        }
    }
}

/* -------------------------------------------------------------------------- */
/*  Main                                                                      */
/* -------------------------------------------------------------------------- */

int main() {
    CHECK_CUDA(cudaSetDevice(1));
    print_device_info(1);

    // --- Parameters (pedagogical example) ---
    const int N = 2;        // minibatch size
    const int C = 3;        // input channels (e.g., RGB)
    const int H = 18;       // input height (→ 16×16 output, divisible by TILE_WIDTH)
    const int W = 18;       // input width
    const int K = 3;        // filter size (3×3)
    const int M = 4;        // output feature maps

    const int H_out = H - K + 1;  // 16
    const int W_out = W - K + 1;  // 16

    const size_t size_X   = N * C * H * W;
    const size_t size_W   = M * C * K * K;
    const size_t size_Y   = N * M * H_out * W_out;

    printf("\n=== Chapter 16: Direct Convolution Forward Kernel (Fig 16.15) ===\n");
    printf("Parameters: N=%d C=%d H=%d W=%d K=%d M=%d\n", N, C, H, W, K, M);
    printf("Output dims: H_out=%d W_out=%d\n", H_out, W_out);
    printf("Input size: %zu floats, Filter size: %zu floats, Output size: %zu floats\n",
           size_X, size_W, size_Y);

    // --- Allocate host memory ---
    float* h_X      = new float[size_X];
    float* h_W      = new float[size_W];
    float* h_Y_cpu  = new float[size_Y];
    float* h_Y_gpu  = new float[size_Y];

    // --- Initialize with known values (small integers for manual verification) ---
    for (size_t i = 0; i < size_X; i++) h_X[i] = static_cast<float>(i % 10);
    for (size_t i = 0; i < size_W; i++) h_W[i] = static_cast<float>((i % 5) + 1) * 0.1f;
    std::memset(h_Y_cpu, 0, size_Y * sizeof(float));
    std::memset(h_Y_gpu, 0, size_Y * sizeof(float));

    // --- CPU reference ---
    printf("\n[1/4] Computing CPU reference...\n");
    gpu_timer cpu_timer;
    cpu_timer.start();
    cpu_conv_forward(N, C, H, W, K, M, h_X, h_W, h_Y_cpu);
    cpu_timer.stop();
    printf("  CPU time: %.3f ms\n", cpu_timer.elapsed_ms());

    // --- Allocate device memory ---
    float *d_X, *d_W, *d_Y;
    CHECK_CUDA(cudaMalloc(&d_X, size_X * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_W, size_W * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_Y, size_Y * sizeof(float)));

    // --- Copy to device ---
    CHECK_CUDA(cudaMemcpy(d_X, h_X, size_X * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_W, h_W, size_W * sizeof(float), cudaMemcpyHostToDevice));

    // --- Launch kernel ---
    int W_grid = (W_out + TILE_WIDTH - 1) / TILE_WIDTH;
    int H_grid = (H_out + TILE_WIDTH - 1) / TILE_WIDTH;
    int T = H_grid * W_grid;

    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 gridDim(M, T, N);

    printf("\n[2/4] Launching kernel...\n");
    printf("  Grid: (%d, %d, %d), Block: (%d, %d, %d), Threads: %d\n",
           gridDim.x, gridDim.y, gridDim.z,
           blockDim.x, blockDim.y, blockDim.z,
           gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y);

    gpu_timer gpu_timer;
    gpu_timer.start();
    conv_forward_kernel<<<gridDim, blockDim>>>(C, H, W, K, W_grid, d_X, d_W, d_Y);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    gpu_timer.stop();
    printf("  GPU kernel time: %.3f ms\n", gpu_timer.elapsed_ms());

    // --- Copy result back ---
    CHECK_CUDA(cudaMemcpy(h_Y_gpu, d_Y, size_Y * sizeof(float), cudaMemcpyDeviceToHost));

    // --- Validate ---
    printf("\n[3/4] Validating...\n");
    bool pass = cpu_allclose(h_Y_cpu, h_Y_gpu, size_Y, 1e-4f);

    // --- Show sample output ---
    printf("\n[4/4] Sample output (first output map, first sample):\n");
    for (int h = 0; h < H_out; h++) {
        printf("  ");
        for (int w = 0; w < W_out; w++) {
            int idx = ((0 * M + 0) * H_out + h) * W_out + w;
            printf("%8.3f ", h_Y_gpu[idx]);
        }
        printf("\n");
    }

    // --- Cleanup ---
    delete[] h_X;
    delete[] h_W;
    delete[] h_Y_cpu;
    delete[] h_Y_gpu;
    CHECK_CUDA(cudaFree(d_X));
    CHECK_CUDA(cudaFree(d_W));
    CHECK_CUDA(cudaFree(d_Y));

    printf("\n=== %s ===\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
