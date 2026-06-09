// Book:      Programming Massively Parallel Processors (Kirk, Hwu & El Hajj, 4th ed.)
// Chapter:   2 — Heterogeneous Data Parallel Computing
// Reference: Figure 2.4
// Concept:   Sequential vector addition — the CPU baseline before parallelization
// Key insight: The for-loop here is what the GPU grid of threads replaces
// Hardware:  GTX 1050, sm_61 (Pascal)
// Compile:   nvcc -std=c++17 -arch=sm_61 -O2 ch02_vec_add_baseline.cu -o vec_add_baseline

#include <cstdio>
#include <cstdlib>
#include <chrono>
#include "../common/cuda_utils.cuh"

// ============================================================
// Sequential vector addition: Figure 2.4
// This is the baseline that we parallelize in subsequent figures
// ============================================================
void vecAdd(float* A_h, float* B_h, float* C_h, int n) {
    for (int i = 0; i < n; ++i) {
        C_h[i] = A_h[i] + B_h[i];
    }
}

// ============================================================
// Main: run sequential version for timing baseline
// ============================================================
int main() {
    const int N = 1 << 20; // 1M elements
    printf("Vector size: %d elements (%.2f MB per array)\n", N, (double)N * sizeof(float) / (1 << 20));

    // Allocate and initialize
    float* A_h = (float*)malloc(N * sizeof(float));
    float* B_h = (float*)malloc(N * sizeof(float));
    float* C_h = (float*)malloc(N * sizeof(float));

    for (int i = 0; i < N; ++i) {
        A_h[i] = (float)(i % 100);
        B_h[i] = (float)((i * 7) % 100);
    }

    // Time the sequential execution
    gpu_timer timer;
    timer.start();
    vecAdd(A_h, B_h, C_h, N);
    timer.stop();
    float cpu_ms = timer.elapsed_ms();

    // Compute metrics
    double gflops = (2.0 * N / 1e9) / (cpu_ms / 1000.0);

    printf("\n--- Sequential Results ---\n");
    printf("CPU time:     %.3f ms\n", cpu_ms);
    printf("GFLOPS:       %.2f\n", gflops);
    printf("Note: This is the serial baseline. GPU version should be compared\n");
    printf("      against this, but note that GPU version includes H2D/D2H\n");
    printf("      transfer overhead which this baseline does not account for.\n");

    // Verify a few elements
    bool spot_check = true;
    for (int i = 0; i < 10 && spot_check; ++i) {
        float expected = A_h[i] + B_h[i];
        if (C_h[i] != expected) {
            printf("MISMATCH at index %d: got %f, expected %f\n", i, C_h[i], expected);
            spot_check = false;
        }
    }
    printf("Spot check:   %s\n", spot_check ? "PASS" : "FAIL");

    free(A_h);
    free(B_h);
    free(C_h);

    return 0;
}
