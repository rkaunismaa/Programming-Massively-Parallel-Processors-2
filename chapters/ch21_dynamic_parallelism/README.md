# Chapter 21 — CUDA Dynamic Parallelism

**Hardware**: GTX 1050, sm_61 (Pascal), 5 SMs, 2 GB VRAM
**Build**: nvcc -std=c++17 -arch=sm_61 -O2 **-rdc=true** -lcudadevrt

## Overview

CUDA Dynamic Parallelism allows GPU kernels to launch other kernels from the
device, enabling recursive algorithms, dynamic work discovery, and adaptive
parallelism without host involvement.

## Kernels Implemented

| Kernel | File | Figure | Description |
|--------|------|--------|-------------|
| Bezier Curves | `ch21_bezier_curves.cu` | 21.7 | Parent discovers work + cudaMalloc, child computes vertices |
| Quadtree | `ch21_quadtree.cu` | 21.10-11 | Recursive spatial subdivision, 4 children per node |
| Exercises | `ch21_exercises.cu` | — | Solutions to all 6 exercises |

## Validation Results

| Kernel | Test | Time | Result |
|--------|------|------|:------:|
| Bezier Curves | 16 lines, max 32 tess points | 28.75 ms | PASS |
| Quadtree | 64 points, max depth 8, min 2 | 34.28 ms | PASS |

## Key DP Concepts Demonstrated

### 1. Parent-Child Kernel Launch (Fig 21.7)
```cpp
__global__ void parent(BezierLine *lines, int n) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < n) {
        // Discover work amount
        lines[i].nVertices = computeTessCount(lines[i]);
        // Allocate memory ON DEVICE
        cudaMalloc(&lines[i].vertexPos, lines[i].nVertices * sizeof(float2));
        // Launch child grid
        child<<<blocks, threads>>>(i, lines, lines[i].nVertices);
    }
}
```

### 2. Device Memory Management
- `cudaMalloc` from device kernel allocates global memory
- Must be freed by `cudaFree` from a device kernel
- Device-allocated memory is device-only (no host cudaMemcpy on sm_61)

### 3. Recursive Launch (Quadtree)
- Each block is one quadtree node
- If node has > min points, last thread launches 4 child blocks
- Recursion continues until min points or max depth reached
- Two point buffers (ping-pong) for reordering

### 4. Configuration
- **Pending launch pool**: `cudaDeviceSetLimit(cudaLimitDevRuntimePendingLaunchCount, N)`
  Default 2048; set to expected launch count to avoid virtualized pool slowdown
- **Streams**: Per-thread named streams enable concurrent child grids
- **Nesting depth**: Max 24 levels (hardware limit)

## Build Requirements

Dynamic parallelism kernels require `-rdc=true` (relocatable device code):
```bash
nvcc -std=c++17 -arch=sm_61 -O2 -rdc=true -o binary source.cu -lcudadevrt
```

## Exercise Solutions

| Ex | Topic | Answer |
|----|-------|--------|
| 1a | Bezier N_LINES=1024, BLOCK_DIM=64 | TRUE: 16 parent threads → 16 child grids |
| 1b | Pool size > default? | FALSE: 1024 fits in default 2048 |
| 1c | Per-thread streams = 16? | FALSE: 16 blocks × 64 threads/block = 1024 streams |
| 2 | Quadtree max depth (64 pts) | (b) 4 (root + 3 subdivision levels) |
| 3 | Total child launches | 84 (4+16+64) — trick: none of the given options |
| 4 | Child inherits __constant__? | TRUE (global/constant memory is shared) |
| 5 | Child accesses parent shared? | FALSE (shared/local are private to block/thread) |
| 6 | Concurrent children (default stream)? | (d) 1 per block (NULL stream serializes) |

## Pitfalls

- **Child kernel MUST be defined before parent** in source file (DP requirement)
- **cudaMalloc from device** creates device-only memory — host cudaMemcpy fails
- **cudaFree from device** is required; host cudaFree doesn't work on device pointers
- **No __syncthreads() after child launch**: cudaDeviceSynchronize() equivalent unavailable in device code; use `cudaStreamSynchronize()` with per-thread streams instead
- **Default NULL stream** serializes all child grids in a block — use named streams for concurrency
