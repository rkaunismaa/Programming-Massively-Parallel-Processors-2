# Chapter 9: Parallel Histogram Implementation Plan

**Goal:** Implement 6 histogram kernels from Chapter 9, covering atomic operations, privatization (global + shared memory), thread coarsening (contiguous + interleaved), and aggregation.

**Kernels:**
- [ ] Task 1: `ch09_basic_histogram.cu` — Fig 9.6: basic atomicAdd on global memory
- [ ] Task 2: `ch09_privatized_global.cu` — Fig 9.9: per-block private copies in global memory
- [ ] Task 3: `ch09_privatized_shared.cu` — Fig 9.10: per-block private copies in shared memory
- [ ] Task 4: `ch09_coarsened_contiguous.cu` — Fig 9.12: coarsening + contiguous partitioning
- [ ] Task 5: `ch09_coarsened_interleaved.cu` — Fig 9.14: coarsening + interleaved partitioning
- [ ] Task 6: `ch09_aggregated.cu` — Fig 9.15: aggregation of consecutive same-bin updates
- [ ] Task 7: README.md with results
- [ ] Task 8: Commit
