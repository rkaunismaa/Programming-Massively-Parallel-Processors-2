/*
 * =============================================================================
 *  ch16_conv_gemm.cu — GEMM-Based Convolutional Layer (Section 16.4)
 * =============================================================================
 *  Book:      Programming Massively Parallel Processors (Kirk, Hwu & El Hajj,
 *             4th Edition, Morgan Kaufmann 2023)
 *  Chapter:   16 — Deep Learning
 *  Section:   16.4 — Formulating a Convolutional Layer as GEMM
 *
 *  Pipeline:
 *    1. Unroll input X[N][C][H][W] → X_unroll[(C*K*K)][(H_out*W_out)]
 *       using the kernel from Fig 16.18.
 *    2. Tiled matrix multiply:
 *         Y[M][W_unroll] = W_filt[M][H_unroll] × X_unroll[H_unroll][W_unroll]
 *       using shared-memory tiling (as in Chapter 5).
 *    3. Validate against the direct convolution CPU reference.
 *
 *  Note: Unrolling is done per-sample to keep memory footprint manageable.
 *        The filter matrix W_filt is used as-is (no duplication needed).
 *
 *  Hardware: GTX 1050, sm_61 (Pascal)
 * =============================================================================
 */

#include "../common/cuda_utils.cuh"
#include <cmath>
#include <cstring>
#include <algorithm>

#define UNROLL_BLOCK 256
#define TILE_SIZE    16

/* -------------------------------------------------------------------------- */
/*  GPU Kernel 1: unroll_Kernel (Fig 16.18)                                   */
/* -------------------------------------------------------------------------- */

__global__ void unroll_kernel(
    int C, int H, int W, int K,
    const float* X, float* X_unroll)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int H_out = H - K + 1;
    int W_out = W - K + 1;
    int W_unroll = H_out * W_out;

    if (t < C * W_unroll) {
        int c         = t / W_unroll;
        int w_unroll  = t % W_unroll;
        int h_out     = w_unroll / W_out;
        int w_out     = w_unroll % W_out;
        int w_base    = c * K * K;

        for (int p = 0; p < K; p++) {
            for (int q = 0; q < K; q++) {
                int h_unroll = w_base + p * K + q;
                int x_idx    = (c * H + (h_out + p)) * W + (w_out + q);
                int u_idx    = h_unroll * W_unroll + w_unroll;
                X_unroll[u_idx] = X[x_idx];
            }
        }
    }
}

/* -------------------------------------------------------------------------- */
/*  GPU Kernel 2: tiled_matmul  (C[M][P] = A[M][Kdim] × B[Kdim][P])          */
/* -------------------------------------------------------------------------- */

__global__ void tiled_matmul_kernel(
    int M, int Kdim, int P,
    const float* A, const float* B, float* C)
{
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int row = by * TILE_SIZE + ty;  // output row (0..M-1)
    int col = bx * TILE_SIZE + tx;  // output col (0..P-1)

    float acc = 0.0f;

    // Loop over tiles of the inner dimension
    for (int t = 0; t < (Kdim + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load A tile: A[row][t*TILE + tx]
        int a_col = t * TILE_SIZE + tx;
        if (row < M && a_col < Kdim)
            As[ty][tx] = A[row * Kdim + a_col];
        else
            As[ty][tx] = 0.0f;

        // Load B tile: B[t*TILE + ty][col]
        int b_row = t * TILE_SIZE + ty;
        if (b_row < Kdim && col < P)
            Bs[ty][tx] = B[b_row * P + col];
        else
            Bs[ty][tx] = 0.0f;

        __syncthreads();

        // Accumulate partial dot product
        for (int k = 0; k < TILE_SIZE; k++) {
            acc += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    if (row < M && col < P) {
        C[row * P + col] = acc;
    }
}

/* -------------------------------------------------------------------------- */
/*  CPU Reference: direct convolution (same as ch16_conv_forward)             */
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
/*  Prepare filter matrix W_filt: M × (C*K*K) — simple reshape, no copy needed*/
/* -------------------------------------------------------------------------- */
// W_weights is already M×C×K×K in row-major, which IS the W_filt layout.
// W_filt[m][c*K*K + p*K + q] == W_weights[m][c][p][q]
// So we can pass W_weights directly as the A matrix.

/* -------------------------------------------------------------------------- */
/*  Main                                                                      */
/* -------------------------------------------------------------------------- */

int main() {
    CHECK_CUDA(cudaSetDevice(1));
    print_device_info(1);

    // --- Parameters (matching textbook Fig 16.16 example) ---
    const int N = 2;        // minibatch
    const int C = 3;        // input channels
    const int H = 12;       // input height
    const int W = 12;       // input width
    const int K = 3;        // filter size
    const int M = 4;        // output feature maps

    const int H_out = H - K + 1;       // 10
    const int W_out = W - K + 1;       // 10
    const int W_unroll = H_out * W_out; // 100 columns in unrolled matrix
    const int H_unroll = C * K * K;     // 27 rows in unrolled matrix

    const size_t size_X        = N * C * H * W;
    const size_t size_unroll   = H_unroll * W_unroll;
    const size_t size_W        = M * H_unroll;     // W_filt: M × (C*K*K)
    const size_t size_Y        = N * M * W_unroll;

    printf("\n=== Chapter 16: GEMM-Based Convolution (Section 16.4) ===\n");
    printf("Parameters: N=%d C=%d H=%d W=%d K=%d M=%d\n", N, C, H, W, K, M);
    printf("Output: H_out=%d W_out=%d  W_unroll=%d  H_unroll=%d\n",
           H_out, W_out, W_unroll, H_unroll);
    printf("Filter matrix: %d×%d  Unrolled: %d×%d  Output: %d×%d\n",
           M, H_unroll, H_unroll, W_unroll, M, W_unroll);

    // --- Allocate host memory ---
    float* h_X      = new float[size_X];
    float* h_W      = new float[size_W];
    float* h_Y_ref  = new float[size_Y];
    float* h_Y_gemm = new float[size_Y];

    // --- Initialize ---
    for (size_t i = 0; i < size_X; i++) h_X[i] = static_cast<float>(i % 20) * 0.5f;
    for (size_t i = 0; i < size_W; i++) h_W[i] = static_cast<float>((i % 7) + 1) * 0.1f;

    // --- CPU reference ---
    printf("\n[1/4] Computing CPU reference (direct convolution)...\n");
    cpu_conv_forward(N, C, H, W, K, M, h_X, h_W, h_Y_ref);

    // --- Allocate device memory ---
    float *d_X, *d_X_unroll, *d_W, *d_Y;
    CHECK_CUDA(cudaMalloc(&d_X, size_X * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_X_unroll, size_unroll * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_W, size_W * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_Y, size_Y * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_X, h_X, size_X * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_W, h_W, size_W * sizeof(float), cudaMemcpyHostToDevice));

    // --- Process each sample ---
    int unroll_threads = C * W_unroll;
    int unroll_grid    = (unroll_threads + UNROLL_BLOCK - 1) / UNROLL_BLOCK;

    dim3 matmul_block(TILE_SIZE, TILE_SIZE, 1);
    dim3 matmul_grid((W_unroll + TILE_SIZE - 1) / TILE_SIZE,
                     (M + TILE_SIZE - 1) / TILE_SIZE, 1);

    printf("\n[2/4] GEMM pipeline (per sample)...\n");
    printf("  Unroll: %d threads, %d blocks\n", unroll_threads, unroll_grid);
    printf("  Matmul grid: (%d, %d) blocks of %d×%d threads\n",
           matmul_grid.x, matmul_grid.y, TILE_SIZE, TILE_SIZE);

    gpu_timer total_timer;
    total_timer.start();

    for (int n = 0; n < N; n++) {
        // Step A: Unroll this sample
        unroll_kernel<<<unroll_grid, UNROLL_BLOCK>>>(
            C, H, W, K,
            d_X + n * C * H * W,   // offset to this sample's X
            d_X_unroll);
        CHECK_CUDA(cudaGetLastError());

        // Step B: Tiled matrix multiply
        //   Y[n] = W_filt[M×H_unroll] × X_unroll[H_unroll×W_unroll]
        tiled_matmul_kernel<<<matmul_grid, matmul_block>>>(
            M, H_unroll, W_unroll,
            d_W, d_X_unroll,
            d_Y + n * M * W_unroll);   // offset to this sample's Y
        CHECK_CUDA(cudaGetLastError());
    }

    CHECK_CUDA(cudaDeviceSynchronize());
    total_timer.stop();
    printf("  Total GPU time (all samples): %.3f ms\n", total_timer.elapsed_ms());

    // --- Copy result ---
    CHECK_CUDA(cudaMemcpy(h_Y_gemm, d_Y, size_Y * sizeof(float), cudaMemcpyDeviceToHost));

    // --- Validate ---
    printf("\n[3/4] Validating GEMM pipeline vs direct convolution...\n");
    bool pass = cpu_allclose(h_Y_ref, h_Y_gemm, size_Y, 1e-3f);

    // --- Show sample output ---
    printf("\n[4/4] Sample output (first output map, first sample):\n");
    for (int h = 0; h < std::min(H_out, 6); h++) {
        printf("  ");
        for (int w = 0; w < std::min(W_out, 6); w++) {
            int idx = ((0 * M + 0) * H_out + h) * W_out + w;
            printf("%8.1f ", h_Y_gemm[idx]);
        }
        if (W_out > 6) printf("...");
        printf("\n");
    }
    if (H_out > 6) printf("  ...\n");

    // --- Expansion ratio ---
    float expansion = (float)(H_unroll * W_unroll) / (float)(C * H * W);
    printf("\nExpansion ratio: %.1f× (theoretical max: K² = %.0f×)\n",
           expansion, (float)(K * K));

    // --- Cleanup ---
    delete[] h_X;
    delete[] h_W;
    delete[] h_Y_ref;
    delete[] h_Y_gemm;
    CHECK_CUDA(cudaFree(d_X));
    CHECK_CUDA(cudaFree(d_X_unroll));
    CHECK_CUDA(cudaFree(d_W));
    CHECK_CUDA(cudaFree(d_Y));

    printf("\n=== %s ===\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
