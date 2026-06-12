/*
 * =============================================================================
 *  ch16_unroll.cu — Input Feature Map Unrolling Kernel (Fig 16.18)
 * =============================================================================
 *  Book:      Programming Massively Parallel Processors (Kirk, Hwu & El Hajj,
 *             4th Edition, Morgan Kaufmann 2023)
 *  Chapter:   16 — Deep Learning
 *  Section:   16.4 — Formulating a Convolutional Layer as GEMM
 *
 *  Unrolls input feature maps X[N][C][H][W] into an expanded matrix
 *  X_unroll[(C*K*K)][(H_out*W_out)] so that convolution becomes a single
 *  matrix multiplication: Y = W_filt × X_unroll.
 *
 *  Each column of X_unroll contains all input pixels needed to compute
 *  one output pixel. Each row segment (K×K rows per channel) corresponds
 *  to one filter weight position across all output pixels.
 *
 *  Thread organization:
 *    Total threads = C * H_out * W_out
 *    Each thread writes K×K elements into one column of X_unroll
 *    1D thread blocks
 *
 *  Hardware: GTX 1050, sm_61 (Pascal)
 * =============================================================================
 */

#include "../common/cuda_utils.cuh"
#include <cmath>
#include <cstring>

#define BLOCK_SIZE 256

/* -------------------------------------------------------------------------- */
/*  GPU Kernel: unroll_Kernel (Fig 16.18)                                     */
/* -------------------------------------------------------------------------- */

__global__ void unroll_kernel(
    int C, int H, int W, int K,
    const float* X, float* X_unroll)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int H_out = H - K + 1;
    int W_out = W - K + 1;
    int W_unroll = H_out * W_out;   // width of unrolled matrix = #output pixels

    if (t < C * W_unroll) {
        // Which input channel this thread works on
        int c = t / W_unroll;
        // Which output pixel (column in X_unroll) this thread fills
        int w_unroll = t % W_unroll;
        // Recover the 2D output pixel coordinate
        int h_out = w_unroll / W_out;
        int w_out = w_unroll % W_out;

        // Starting row in X_unroll for channel c
        int w_base = c * K * K;

        // Gather K×K input patch and write into X_unroll column
        for (int p = 0; p < K; p++) {
            for (int q = 0; q < K; q++) {
                int h_unroll = w_base + p * K + q;
                // X[c, h_out+p, w_out+q]  (single sample, omit batch dim here)
                int x_idx = (c * H + (h_out + p)) * W + (w_out + q);
                // X_unroll[h_unroll, w_unroll]
                int u_idx = h_unroll * W_unroll + w_unroll;
                X_unroll[u_idx] = X[x_idx];
            }
        }
    }
}

/* -------------------------------------------------------------------------- */
/*  CPU Reference: sequential unrolling                                       */
/* -------------------------------------------------------------------------- */

void cpu_unroll(
    int C, int H, int W, int K,
    const float* X, float* X_unroll)
{
    int H_out = H - K + 1;
    int W_out = W - K + 1;
    int W_unroll = H_out * W_out;

    std::memset(X_unroll, 0, C * K * K * W_unroll * sizeof(float));

    for (int c = 0; c < C; c++) {
        int w_base = c * K * K;
        for (int p = 0; p < K; p++) {
            for (int q = 0; q < K; q++) {
                for (int h_out = 0; h_out < H_out; h_out++) {
                    for (int w_out = 0; w_out < W_out; w_out++) {
                        int h_unroll = w_base + p * K + q;
                        int w_unroll = h_out * W_out + w_out;
                        int x_idx = (c * H + (h_out + p)) * W + (w_out + q);
                        int u_idx = h_unroll * W_unroll + w_unroll;
                        X_unroll[u_idx] = X[x_idx];
                    }
                }
            }
        }
    }
}

/* -------------------------------------------------------------------------- */
/*  Verify unrolling: reconstruct convolution via matrix multiply              */
/* -------------------------------------------------------------------------- */

// Simple CPU matrix multiply: C[M][P] = A[M][K] × B[K][P]
void cpu_matmul(int M, int Kdim, int P,
                const float* A, const float* B, float* C)
{
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < P; j++) {
            float acc = 0.0f;
            for (int k = 0; k < Kdim; k++) {
                acc += A[i * Kdim + k] * B[k * P + j];
            }
            C[i * P + j] = acc;
        }
    }
}

/* -------------------------------------------------------------------------- */
/*  Main                                                                      */
/* -------------------------------------------------------------------------- */

int main() {
    CHECK_CUDA(cudaSetDevice(1));
    print_device_info(1);

    // --- Parameters ---
    const int C = 3;        // input channels
    const int H = 5;        // input height
    const int W = 5;        // input width
    const int K = 2;        // filter size
    const int M = 2;        // output feature maps

    const int H_out = H - K + 1;  // 4
    const int W_out = W - K + 1;  // 4
    const int W_unroll = H_out * W_out;                     // 16 columns
    const int H_unroll = C * K * K;                         // 12 rows

    const size_t size_X       = C * H * W;                   // 75
    const size_t size_unroll  = H_unroll * W_unroll;         // 192
    const size_t size_W_filt  = M * C * K * K;               // 24
    const size_t size_Y       = M * H_out * W_out;           // 32

    printf("\n=== Chapter 16: Input Unrolling Kernel (Fig 16.18) ===\n");
    printf("Parameters: C=%d H=%d W=%d K=%d M=%d\n", C, H, W, K, M);
    printf("Output dims: H_out=%d W_out=%d\n", H_out, W_out);
    printf("Unrolled matrix: %d rows × %d columns (%zu floats)\n",
           H_unroll, W_unroll, size_unroll);

    // --- Allocate host memory ---
    float* h_X        = new float[size_X];
    float* h_U_cpu    = new float[size_unroll];
    float* h_U_gpu    = new float[size_unroll];
    float* h_W_filt   = new float[size_W_filt];
    float* h_Y_cpu    = new float[size_Y];
    float* h_Y_gpu    = new float[size_Y];

    // --- Initialize with known values ---
    for (size_t i = 0; i < size_X; i++)      h_X[i]      = static_cast<float>(i);
    for (size_t i = 0; i < size_W_filt; i++)  h_W_filt[i] = static_cast<float>((i % 4) + 1) * 0.1f;

    // --- CPU reference: unroll + verify ---
    printf("\n[1/4] Computing CPU unroll reference...\n");
    cpu_unroll(C, H, W, K, h_X, h_U_cpu);

    // --- Allocate device memory ---
    float *d_X, *d_X_unroll;
    CHECK_CUDA(cudaMalloc(&d_X, size_X * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_X_unroll, size_unroll * sizeof(float)));

    // --- Copy to device ---
    CHECK_CUDA(cudaMemcpy(d_X, h_X, size_X * sizeof(float), cudaMemcpyHostToDevice));

    // --- Launch unroll kernel ---
    int total_threads = C * W_unroll;
    int grid_size = (total_threads + BLOCK_SIZE - 1) / BLOCK_SIZE;

    printf("\n[2/4] Launching unroll kernel...\n");
    printf("  Threads: %d, Grid: %d, Block: %d\n", total_threads, grid_size, BLOCK_SIZE);

    gpu_timer gpu_timer;
    gpu_timer.start();
    unroll_kernel<<<grid_size, BLOCK_SIZE>>>(C, H, W, K, d_X, d_X_unroll);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    gpu_timer.stop();
    printf("  GPU kernel time: %.3f ms\n", gpu_timer.elapsed_ms());

    // --- Copy result back ---
    CHECK_CUDA(cudaMemcpy(h_U_gpu, d_X_unroll, size_unroll * sizeof(float), cudaMemcpyDeviceToHost));

    // --- Validate unrolled matrix ---
    printf("\n[3/4] Validating unrolled matrix...\n");
    bool unroll_pass = cpu_allclose(h_U_cpu, h_U_gpu, size_unroll, 1e-4f);

    // --- Verify GEMM equivalence ---
    printf("\n[4/4] Verifying GEMM equivalence...\n");
    // CPU conv reference
    float* h_Y_ref = new float[size_Y];
    for (int m = 0; m < M; m++) {
        for (int h = 0; h < H_out; h++) {
            for (int w = 0; w < W_out; w++) {
                float acc = 0.0f;
                for (int c = 0; c < C; c++) {
                    for (int p = 0; p < K; p++) {
                        for (int q = 0; q < K; q++) {
                            int x_idx = (c * H + (h + p)) * W + (w + q);
                            int w_idx = ((m * C + c) * K + p) * K + q;
                            acc += h_X[x_idx] * h_W_filt[w_idx];
                        }
                    }
                }
                h_Y_ref[m * W_unroll + h * W_out + w] = acc;
            }
        }
    }

    // GEMM via unrolled matrix: Y[M][W_unroll] = W_filt[M][H_unroll] × U_gpu[H_unroll][W_unroll]
    cpu_matmul(M, H_unroll, W_unroll, h_W_filt, h_U_gpu, h_Y_gpu);

    bool gemm_pass = cpu_allclose(h_Y_ref, h_Y_gpu, size_Y, 1e-3f);

    // --- Show sample ---
    printf("\nSample: X_unroll first 3 columns (from GPU):\n");
    for (int r = 0; r < H_unroll; r++) {
        printf("  row %2d:", r);
        for (int c = 0; c < 3 && c < W_unroll; c++) {
            printf(" %8.1f", h_U_gpu[r * W_unroll + c]);
        }
        printf("\n");
    }

    printf("\nSample: Y from GEMM (first output map):\n");
    for (int h = 0; h < H_out; h++) {
        printf("  ");
        for (int w = 0; w < W_out; w++) {
            printf("%8.1f ", h_Y_gpu[0 * W_unroll + h * W_out + w]);
        }
        printf("\n");
    }

    // --- Cleanup ---
    delete[] h_X;
    delete[] h_U_cpu;
    delete[] h_U_gpu;
    delete[] h_W_filt;
    delete[] h_Y_cpu;
    delete[] h_Y_gpu;
    delete[] h_Y_ref;
    CHECK_CUDA(cudaFree(d_X));
    CHECK_CUDA(cudaFree(d_X_unroll));

    bool pass = unroll_pass && gemm_pass;
    printf("\n=== Unroll: %s | GEMM equivalence: %s | OVERALL: %s ===\n",
           unroll_pass ? "PASS" : "FAIL",
           gemm_pass   ? "PASS" : "FAIL",
           pass        ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
