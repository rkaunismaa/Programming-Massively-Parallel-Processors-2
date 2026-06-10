# Chapter 9: Parallel Histogram (Figures 9.6, 9.9, 9.10, 9.12, 9.14, 9.15)

**GTX 1050** | sm_61 Pascal, 5 SMs, 1455 MHz, ~2 GB VRAM

## Overview

This chapter introduces **parallel histogram computation** — a pattern where output elements can be updated by *any* thread, making it impossible to apply the owner-computes rule. The fundamental tool for correctness is the **atomic operation**, but naive use leads to severe contention. Six progressively optimized kernels demonstrate the evolution from basic atomic operations through privatization, coarsening, and aggregation.

All kernels compute a 7-bin text histogram from a 16M-character input string (biased distribution toward middle letters to create contention).

## Kernels Implemented

| # | File | Figure | Concept | Time (ms) | Atomics/sec | Speedup | Status |
|---|------|--------|---------|-----------|-------------|---------|--------|
| 1 | `ch09_basic_histogram` | Fig 9.6 | Basic atomicAdd on global memory | 8.20 | 2,047 M | 1.0× | ✅ |
| 2 | `ch09_privatized_global` | Fig 9.9 | Per-block private copy in global mem | 4.84 | 3,470 M | 1.7× | ✅ |
| 3 | `ch09_privatized_shared` | Fig 9.10 | Per-block private copy in shared mem | 2.12 | 7,902 M | 3.9× | ✅ |
| 4 | `ch09_coarsened_contiguous` | Fig 9.12 | Coarsening + contiguous partitioning | 1.88 | 8,919 M | 4.4× | ✅ |
| 5 | `ch09_coarsened_interleaved` | Fig 9.14 | Coarsening + interleaved partitioning | **1.09** | **15,357 M** | **7.5×** | ✅ |
| 6 | `ch09_aggregated` | Fig 9.15 | Aggregation (streaky data optimization) | 1.35 | 12,390 M | 6.1× | ✅ |

## Key Concepts

### Atomic Operations (Fig 9.6)
- **Read-modify-write race condition**: two threads reading the same memory location, modifying, and writing — one thread's update can be lost
- **`atomicAdd`** ensures the read-modify-write sequence is indivisible (serializes concurrent updates to the same location)
- **Problem**: if many threads target the same bin, atomic operations serialize → throughput plummets
- For 16M elements on 7 bins with biased distribution: only **2,047 M atomics/sec** on GTX 1050

### Privatization — Global Memory (Fig 9.9)
- **Idea**: each block gets its own private copy of the histogram
- Blocks update their private copies → contention reduced from all-threads to per-block
- After computation, private copies are merged atomically into the public copy
- **Result**: 1.7× faster (4.84 ms) — contention drops from 65K blocks to ~5 SMs

### Privatization — Shared Memory (Fig 9.10)
- **Idea**: private histogram stored in `__shared__` memory instead of global
- Shared memory atomics have much lower latency (tens of cycles vs hundreds)
- **Result**: 3.9× faster (2.12 ms) — the single biggest optimization

### Thread Coarsening — Contiguous (Fig 9.12)
- Each thread processes CFACTOR=4 contiguous elements: `[tid*4 .. tid*4+3]`
- Reduces number of blocks → fewer private copies to merge
- **Problem**: contiguous partitioning means threads in a warp access distant addresses → poor memory coalescing
- **Result**: 4.4× faster (1.88 ms) — merge overhead reduced, but coalescing costs us

### Thread Coarsening — Interleaved (Fig 9.14)
- Each thread processes CFACTOR=4 elements with stride = total_threads
- Thread 0 processes elements 0, stride, 2*stride, ...
- Thread 1 processes elements 1, stride+1, 2*stride+1, ...
- **Consecutive threads access consecutive addresses** → full memory coalescing
- **Result**: 7.5× faster (1.09 ms) — best overall for this dataset

### Aggregation (Fig 9.15)
- **Idea**: merge consecutive same-bin updates into a single atomicAdd
- Each thread keeps an accumulator + prevBinIdx
- When bin changes → flush accumulator; when same bin → increment accumulator
- **Trade-off**: beneficial for streaky data (e.g., sky images with large uniform regions), but extra code overhead when streaks are rare
- **Result**: 6.1× faster (1.35 ms) — overhead exceeds benefit on this random-ish data

## Performance Analysis

```
Speedup progression (log scale):
  1.0× ─ Basic (8.20 ms)
  1.7× ─ Privatized global (4.84 ms)
  3.9× ─ Privatized shared (2.12 ms)
  4.4× ─ +Contiguous coarsening (1.88 ms)
  7.5× ─ +Interleaved coarsening (1.09 ms) ★ Best
  6.1× ─ +Aggregation (1.35 ms) [streak-dependent]
```

## Data Generation

The test input uses biased random text where:
- 60% of characters come from the middle bins (`i` through `r`)
- 25% from early letters (`a` through `h`)
- 15% from late letters (`s` through `z`)

This mimics the letter distribution of real text and creates significant contention on the middle histogram bins.

## Compile Commands

```bash
cd PMPP/chapters/ch09_histogram
nvcc -std=c++17 -arch=sm_61 -O2 -o ch09_basic_histogram ch09_basic_histogram.cu
nvcc -std=c++17 -arch=sm_61 -O2 -o ch09_privatized_global ch09_privatized_global.cu
nvcc -std=c++17 -arch=sm_61 -O2 -o ch09_privatized_shared ch09_privatized_shared.cu
nvcc -std=c++17 -arch=sm_61 -O2 -o ch09_coarsened_contiguous ch09_coarsened_contiguous.cu
nvcc -std=c++17 -arch=sm_61 -O2 -o ch09_coarsened_interleaved ch09_coarsened_interleaved.cu
nvcc -std=c++17 -arch=sm_61 -O2 -o ch09_aggregated ch09_aggregated.cu
```

## Reference

All kernels validated against a CPU sequential reference (`cpu_histogram`). Input: 16M characters, 7 histogram bins (4 letters per bin, last bin: 2 letters).
