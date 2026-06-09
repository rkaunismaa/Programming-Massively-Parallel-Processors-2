# Chapter 6: Performance Considerations

## Overview

This chapter covers the key performance optimization techniques for CUDA applications. It consolidates the architectural knowledge from Chapters 2-5 into actionable optimization strategies that every CUDA programmer should apply.

## Key Concepts

### 6.1 Memory Coalescing
When threads in a warp execute load/store instructions simultaneously, the hardware combines (coalesces) accesses to consecutive memory locations into a single DRAM transaction. This is the single most important memory optimization.

- **Coalesced access**: consecutive threads access consecutive memory addresses (stride-1)
- **Uncoalesced access**: consecutive threads access strided addresses (e.g., stride-N)
- **Impact**: uncoalesced access can degrade bandwidth utilization by up to 32x (one warp = 32 threads)

### 6.2 Hiding Memory Latency
DRAM systems use multiple banks and channels to hide access latency. Key principles:

- **Banks**: multiple DRAM banks connected to each channel allow overlapping access latency
- **Channels**: each channel is a memory controller with its own bus
- **Interleaved distribution**: hardware spreads array elements across banks/channels automatically
- **Occupancy matters**: more resident threads = more concurrent memory requests = better latency hiding

### 6.3 Thread Coarsening
Assign each thread multiple units of work instead of one. Benefits:
- Reduces redundant data loading when thread blocks would be serialized anyway
- Improves arithmetic intensity by reusing data already in registers/shared memory
- Trade-off: reduces exposed parallelism, so tune COARSE_FACTOR per device

### 6.4 Corner Turning
Technique for coalescing accesses to column-major matrices. When loading a tile from a column-major matrix, swap `threadIdx.x` and `threadIdx.y` roles so consecutive threads access consecutive memory locations. The shared memory tile ends up transposed, but the dot product still computes correctly because both indices swap symmetrically.

### 6.5 Optimization Checklist (Table 6.1)
1. Maximize occupancy (threads per SM)
2. Minimize control divergence within warps
3. Ensure memory coalescing
4. Use shared memory for data reuse (tiling)
5. Minimize global memory traffic
6. Apply thread coarsening when beneficial
7. Use appropriate data types (half precision when possible)
8. Use constant memory for read-only broadcast data

## Hardware: GTX 1050 (sm_61)
- 5 SMs, 2 GB VRAM
- Shared memory per SM: 48 KB
- Registers per SM: 65,536
- Max threads per block: 1024
- Memory bus width: 128-bit
- All kernels explicitly target device 1 via `cudaSetDevice(1)`

## Code Examples

### ch06_thread_coarsening.cu
Thread-coarsened tiled matrix multiplication (Fig 6.13). Each thread computes COARSE_FACTOR=4 output elements using register accumulation. Grid is shrunk in x-dimension by COARSE_FACTOR.

### ch06_corner_turning.cu
Tiled matrix multiplication where matrix B is stored in column-major layout. Uses corner turning to ensure coalesced global memory accesses when loading B tiles into shared memory.

### ch06_memory_coalescing_demo.cu
Benchmark comparing coalesced vs uncoalesced memory access patterns. Demonstrates the dramatic performance difference between stride-1 and stride-32 access patterns.

## Performance Results (GTX 1050, sm_61)

| Kernel | Matrix Size | Time | GFLOPS | Speedup | Validation |
|---|---|---|---|---|---|
| Thread Coarsening (COARSE=4) | 2048x2048 | 73.10 ms | 235.01 | - | PASSED |
| Corner Turning (col-major B) | 2048x2048 | 96.63 ms | 177.79 | - | PASSED |
| Memory Coalescing (coalesced) | 1M elements | 0.093 ms | 86.3 GB/s | 9.47x | PASSED |
| Memory Coalescing (uncoalesced) | 1M elements | 0.878 ms | 9.1 GB/s | baseline | PASSED |

### Key Observations
- Thread coarsening achieves 235 GFLOPS -- excellent for GTX 1050 (theoretical peak ~1.1 TFLOPS for FP32)
- Corner turning adds overhead from the transposed shared memory access pattern, yielding 178 GFLOPS
- Memory coalescing benchmark shows 9.47x speedup for coalesced vs uncoalesced access
- Theoretical maximum speedup for coalescing is 32x (one warp), but memory-level parallelism reduces the gap

## Comparison with Chapter 5
Chapter 5 tiled matmul (static, TILE_WIDTH=16) achieved 265 GFLOPS on the same hardware. The thread coarsening variant (TILE_WIDTH=32, COARSE=4) achieves 235 GFLOPS -- slightly lower due to the larger tile size and register pressure from maintaining 4 Pvalue accumulators.
