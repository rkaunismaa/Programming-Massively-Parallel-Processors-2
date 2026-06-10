# Chapter 10 — Parallel Reduction & Minimizing Divergence

**Figures 10.6, 10.9, 10.11, 10.13, 10.15** — Five progressively optimized sum reduction kernels.

## Hardware

| GPU | CC | SMs | Shared/block | Max threads/block |
|-----|----|-----|-------------|-------------------|
| GTX 1050 (device 1) | 6.1 (Pascal) | 5 | 48 KB | 1024 |

## Kernels

### Fig 10.6 — Simple Sum Reduction (Interleaved, Stride-Doubling)
`ch10_simple_reduction.cu`

Naive parallel reduction tree. Each thread owns position `2*threadIdx.x` (even locations). Stride doubles each iteration (1, 2, 4, ...). Active threads determined by `threadIdx.x % stride == 0`.

**Limitations:**
- Severe control divergence: by iteration 5 only 1/32 threads per warp active (35% resource utilization per the book's analysis)
- Poor memory coalescing: adjacent threads access locations 2 elements apart
- One block only: max 2048 elements

### Fig 10.9 — Convergent Reduction (Stride-Halving)
`ch10_convergent_reduction.cu`

Improved assignment: owner positions are consecutive (`threadIdx.x`). Stride starts at `blockDim.x` and halves each iteration. Active threads: `threadIdx.x < stride`.

**Advantages:**
- Entire warps become inactive (no divergence within active warps)
- ~66% resource utilization (vs ~35% for Fig 10.6)
- Memory coalesced (adjacent threads access adjacent memory)
- 3.9× fewer global memory requests than Fig 10.6

### Fig 10.11 — Shared Memory Reduction
`ch10_shared_memory_reduction.cu`

First iteration absorbed into initial global→shared load: each thread loads `input[t] + input[t + blockDim.x]` directly into shared memory. All subsequent iterations operate on shared memory only.

**Advantages:**
- Only N+1 global memory accesses (N reads + 1 output write)
- vs ~3N/2 for global-memory version (Fig 10.9)
- Input array is NOT modified (preserved for later use)
- Lower latency from shared memory

### Fig 10.13 — Multiblock Segmented Reduction
`ch10_multiblock_reduction.cu`

Extends shared-memory reduction to arbitrary input sizes. Partitions input into segments of `2 × blockDim.x` elements per block. Each block independently sums its segment in shared memory, then thread 0 atomically adds to global output.

**Key features:**
- Works for millions/billions of elements
- Atomic add ensures correctness across blocks
- 256 blocks × 512 elements = 131,072 elements per run

### Fig 10.15 — Thread-Coarsened Reduction
`ch10_coarsened_reduction.cu` (CF=4)

Each thread independently accumulates COARSE_FACTOR pairs into a register before entering the reduction tree. Fewer blocks → less hardware underutilization during final tree iterations.

**Key advantages (per the book):**
- Fewer total steps (6 vs 8 for CF=2 vs 2× uncoarsened)
- More steps at full utilization (3 vs 2)
- Fewer underutilized barrier+shared-memory steps (3 vs 6)
- Coarsening factor can be tuned per device

## Results (GTX 1050)

| Kernel | Input | Blocks×Threads | Time | Bandwidth | Validation |
|--------|-------|----------------|------|-----------|------------|
| Fig 10.6 — Simple | 2,048 | 1 × 1,024 | 0.011 ms | 1.45 GB/s | PASSED |
| Fig 10.9 — Convergent | 2,048 | 1 × 1,024 | 0.009 ms | 1.78 GB/s | PASSED (exact) |
| Fig 10.11 — Shared mem | 2,048 | 1 × 1,024 | 0.008 ms | 1.00 GB/s | PASSED (exact) |
| Fig 10.13 — Multiblock | 131,072 | 256 × 256 | 0.024 ms | 22.17 GB/s | PASSED |
| Fig 10.15 — Coarsened (CF=4) | 262,144 | 128 × 256 | 0.025 ms | 42.45 GB/s | PASSED |

> **Note:** Single-block kernel times are dominated by launch overhead at these tiny input sizes. The convergent kernel is exact because float addition is associative at this scale; the simple kernel has a tiny FP rounding difference. Multiblock and coarsened kernels show meaningful bandwidth on larger inputs.

## Key Concepts

- **Parallel reduction tree**: log₂(N) rounds, each halving the active work
- **Control divergence**: interleaved addressing (Fig 10.6) wastes 65%+ of execution resources
- **Convergent addressing** (Fig 10.9): consecutive owner positions keep warps divergence-free
- **Memory coalescing**: convergent addressing also provides coalesced global memory accesses
- **Shared memory** (Fig 10.11): eliminates all intermediate global memory traffic
- **Segmented multiblock** (Fig 10.13): atomic operations for cross-block accumulation
- **Thread coarsening** (Fig 10.15): reduces parallelization overhead by serializing work within threads
