/*
 * =============================================================================
 *  CUDA Utilities Header — Shared across all PMPP code examples
 * =============================================================================
 *  Book:      Programming Massively Parallel Processors (Kirk, Hwu & El Hajj,
 *             4th Edition, Morgan Kaufmann 2023)
 *  Purpose:   Common macros, types, and helper functions used throughout
 *             all chapter code files.
 *  Hardware:  GTX 1050, sm_61 (Pascal) — all code targets this GPU
 * =============================================================================
 */

#ifndef CUDA_UTILS_CUH
#define CUDA_UTILS_CUH

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <string>
#include <cuda_runtime.h>
#ifdef USE_CUBLAS
#include <cublas_v2.h>
#endif
#ifdef USE_CUDNN
#include <cudnn.h>
#endif

/* -------------------------------------------------------------------------- */
/*  CUDA Error Checking                                                        */
/* -------------------------------------------------------------------------- */

#define CHECK_CUDA(call)                                                         \
    do {                                                                         \
        cudaError_t err = call;                                                  \
        if (err != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error at %s:%d: %s (%s)\n",                    \
                    __FILE__, __LINE__,                                           \
                    cudaGetErrorString(err), #call);                              \
            exit(EXIT_FAILURE);                                                  \
        }                                                                        \
    } while (0)

#ifdef USE_CUBLAS
#define CHECK_CUBLAS(call)                                                       \
    do {                                                                         \
        cublasStatus_t status = call;                                            \
        if (status != CUBLAS_STATUS_SUCCESS) {                                   \
            fprintf(stderr, "cuBLAS error at %s:%d: %d (%s)\n",                  \
                    __FILE__, __LINE__, (int)status, #call);                      \
            exit(EXIT_FAILURE);                                                  \
        }                                                                        \
    } while (0)
#endif

#ifdef USE_CUDNN
#define CHECK_CUDNN(call)                                                        \
    do {                                                                         \
        cudnnStatus_t status = call;                                             \
        if (status != CUDNN_STATUS_SUCCESS) {                                    \
            fprintf(stderr, "cuDNN error at %s:%d: %s (%s)\n",                   \
                    __FILE__, __LINE__,                                           \
                    cudnnGetErrorString(status), #call);                          \
            exit(EXIT_FAILURE);                                                  \
        }                                                                        \
    } while (0)
#endif

/* -------------------------------------------------------------------------- */
/*  GPU Timer — wraps cudaEvent_t for kernel timing                            */
/* -------------------------------------------------------------------------- */

struct gpu_timer {
    cudaEvent_t start_event;
    cudaEvent_t stop_event;

    gpu_timer() {
        CHECK_CUDA(cudaEventCreate(&start_event));
        CHECK_CUDA(cudaEventCreate(&stop_event));
    }

    ~gpu_timer() {
        cudaEventDestroy(start_event);
        cudaEventDestroy(stop_event);
    }

    void start() {
        CHECK_CUDA(cudaEventRecord(start_event, 0));
    }

    void stop() {
        CHECK_CUDA(cudaEventRecord(stop_event, 0));
        CHECK_CUDA(cudaEventSynchronize(stop_event));
    }

    float elapsed_ms() const {
        float ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start_event, stop_event));
        return ms;
    }
};

/* -------------------------------------------------------------------------- */
/*  Device Info — prints GPU properties at runtime                             */
/* -------------------------------------------------------------------------- */

inline void print_device_info(int device_id = -1) {
    int count;
    CHECK_CUDA(cudaGetDeviceCount(&count));

    if (device_id < 0) device_id = 0;
    if (device_id >= count) {
        fprintf(stderr, "Device %d not available (%d devices)\n", device_id, count);
        exit(EXIT_FAILURE);
    }

    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device_id));

    printf("\n=== Device %d: %s ===\n", device_id, prop.name);
    printf("  Compute capability   : %d.%d\n", prop.major, prop.minor);
    printf("  Multi-processors     : %d SM(s)\n", prop.multiProcessorCount);
    printf("  Clock rate           : %d MHz\n", prop.clockRate / 1000);
    printf("  Global memory        : %zu MB\n", prop.totalGlobalMem / (1024 * 1024));
    printf("  Shared mem / block   : %zu KB\n", prop.sharedMemPerBlock / 1024);
    printf("  Registers / block    : %d\n", prop.regsPerBlock);
    printf("  Threads / block      : %d\n", prop.maxThreadsPerBlock);
    printf("  Warp size            : %d\n", prop.warpSize);
    printf("  Max dims (x,y,z)     : (%d, %d, %d)\n",
           prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("  Max grid size (x,y,z): (%d, %d, %d)\n",
           prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("  Memory bus width     : %d-bit\n", prop.memoryBusWidth);
    printf("  ECC enabled          : %s\n", prop.ECCEnabled ? "yes" : "no");
    printf("\n");
}

/* -------------------------------------------------------------------------- */
/*  CPU Reference Validation — element-wise float comparison                   */
/* -------------------------------------------------------------------------- */

inline bool cpu_allclose(const float *a, const float *b, int n, float tol = 1e-4f) {
    bool ok = true;
    int first_mismatch = -1;

    for (int i = 0; i < n; i++) {
        float diff = std::fabs(a[i] - b[i]);
        // Relative tolerance: scale by magnitude of a[i] when nonzero
        float rel_tol = tol * std::fmaxf(1.0f, std::fabs(a[i]));
        if (diff > rel_tol) {
            if (first_mismatch < 0) {
                first_mismatch = i;
            }
            ok = false;
        }
    }

    if (!ok) {
        printf("  FAIL: first mismatch at index %d: expected %.6f, got %.6f\n",
               first_mismatch, a[first_mismatch], b[first_mismatch]);
        // Show up to 5 mismatches for context
        int shown = 0;
        for (int i = 0; i < n && shown < 5; i++) {
            float diff = std::fabs(a[i] - b[i]);
            float rel_tol = tol * std::fmaxf(1.0f, std::fabs(a[i]));
            if (diff > rel_tol) {
                printf("    index %d: expected %.6f, got %.6f (diff=%.6e)\n",
                       i, a[i], b[i], diff);
                shown++;
            }
        }
    }

    return ok;
}

/* -------------------------------------------------------------------------- */
/*  Matrix Validation — 2D float comparison (row-major)                       */
/* -------------------------------------------------------------------------- */

inline bool cpu_allclose_matrix(const float *a, const float *b,
                                int rows, int cols, float tol = 1e-3f) {
    bool ok = true;
    int mismatches = 0;

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            int idx = i * cols + j;
            float diff = std::fabs(a[idx] - b[idx]);
            float rel_tol = tol * std::fmaxf(1.0f, std::fabs(a[idx]));
            if (diff > rel_tol) {
                if (mismatches < 5) {
                    printf("  FAIL: element [%d,%d] = %.6f (expected %.6f, diff=%.6e)\n",
                           i, j, b[idx], a[idx], diff);
                }
                mismatches++;
                ok = false;
            }
        }
    }

    if (mismatches > 5) {
        printf("  ... and %d more mismatches\n", mismatches - 5);
    }

    return ok;
}

/* -------------------------------------------------------------------------- */
/*  Alignment helpers                                                          */
/* -------------------------------------------------------------------------- */

inline int round_up(int a, int multiple) {
    return ((a + multiple - 1) / multiple) * multiple;
}

inline size_t round_up_size(size_t a, size_t multiple) {
    return ((a + multiple - 1) / multiple) * multiple;
}

#endif /* CUDA_UTILS_CUH */
