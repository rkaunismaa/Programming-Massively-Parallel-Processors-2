# Chapter 5: Memory Architecture and Data Locality

## Overview

This chapter covers the GPU memory hierarchy and how to use shared memory to dramatically improve performance through data tiling.

## Key Concepts

### Memory Hierarchy
- **Registers**: Fastest, per-thread, on-chip. Low latency, high bandwidth.
- **Shared Memory**: Fast, per-block, on-chip. Used for data sharing within a block.
- **Local Memory**: Per-thread, actually in global memory. Used for spilled registers and automatic arrays.
- **Global Memory**: Slowest, entire grid, off-chip DRAM. High latency, limited bandwidth.
- **Constant Memory**: Read-only, cached, entire grid scope.

### Memory Types Table (Table 5.1)
| Declaration | Memory | Scope | Lifetime |
|---|---|---|---|
| Automatic scalar vars | Register | Thread | Grid |
| Automatic array vars | Local | Thread | Grid |
| `__shared__` | Shared | Block | Grid |
| `__device__` | Global | Grid | Application |
| `__constant__` | Constant | Grid | Application |

### Tiling
The main optimization technique: load data from global memory into shared memory tiles, then reuse those tiles multiple times before loading the next tile. This dramatically reduces global memory traffic.

### Compute-to-Memory Access Ratio
- Naive matmul: 2 FLOP / 8 bytes = 0.25 FLOP/B (memory-bound)
- To achieve peak performance: need ~12.5 FLOP/B
- Tiling increases this ratio by reusing data loaded into shared memory

### Roofline Model
Visual model showing arithmetic intensity (x-axis) vs computational throughput (y-axis). Applications are either memory-bound or compute-bound depending on their position relative to hardware limits.

## Hardware: GTX 1050 (sm_61)
- 5 SMs, 2 GB VRAM
- Shared memory per SM: 48 KB
- Registers per SM: 32,768
- Max threads per block: 1024
- Max blocks per SM: 8
- All kernels explicitly target device 1 via `cudaSetDevice(1)`

## Code Examples

### ch05_tiled_matmul_static.cu
Tiled matrix multiplication using static shared memory (compile-time TILE_WIDTH).

### ch05_tiled_matmul_dynamic.cu
Tiled matrix multiplication using dynamic shared memory (runtime tile size).

### ch05_tiled_matmul_boundary.cu
Tiled matrix multiplication with boundary checks for non-square and non-divisible matrices.

## Exercises
See EXERCISES_SOLUTIONS.md

## Performance Comparison (GTX 1050, sm_61)
| Kernel | Time | GFLOPS | Notes |
|---|---|---|---|
| Tiled matmul (static) | 517.84 ms | 265.41 | Shared memory tiling |
| Tiled matmul (dynamic) | 682.81 ms | 201.29 | Runtime tile size |
| Tiled + boundary | 553.65 ms | 248.42 | Non-divisible matrices |

Note: Ch3 naive matmul baseline (366.53 GFLOPS) was measured on RTX 4090. Tiling still provides massive improvement over naive approaches on the 1050.
