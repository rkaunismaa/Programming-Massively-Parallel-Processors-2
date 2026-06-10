# Chapter 11 — Prefix Sum (Scan)

**An introduction to work efficiency in parallel algorithms**

| Figures 11.3, 11.7 | Four progressively optimized parallel scan kernels |
|---|---|

## Hardware

| GPU | CC | SMs | Shared/block | Max threads/block |
|-----|----|-----|-------------|-------------------|
| GTX 1050 (device 1) | 6.1 (Pascal) | 5 | 48 KB | 1024 |

## Kernels

### Fig 11.3 — Kogge-Stone Inclusive Scan (Segmented)

`ch11_kogge_stone_scan.cu`

Per-block segmented scan using the Kogge-Stone adder design. Each block independently scans its SECTION_SIZE-element segment using a stride-doubling tree pattern.

**Key design points:**
- Each thread owns one element, all threads in a block collaborate
- Stride doubles each iteration (1, 2, 4, ..., blockDim.x)
- **Temp variable + double `__syncthreads()`** required to avoid write-after-read race condition (unlike reduction — because updated values ARE read by other threads in the same iteration)
- Work complexity: O(N log N) — **not work-efficient** (vs sequential O(N))
- Control divergence occurs only in the first warp when stride < 32

**Race condition (write-after-read):** Active thread i reads XY[i - stride] (written by thread i - stride in the same iteration), so thread i - stride must not overwrite XY[i - stride] before thread i reads it. Solved by: (1) read into private `temp`, (2) barrier, (3) write from `temp`.

### Fig 11.7 — Brent-Kung Inclusive Scan (Segmented)

`ch11_brent_kung_scan.cu`

Uses a **reduction tree phase** (N-1 ops) + **reverse tree phase** (N - 1 - log₂(N) ops) for total work **2N - 2 - log₂(N) = O(N)** — work-efficient!

**Key design points:**
- Each block has SECTION_SIZE/2 threads (each handles 2 elements)
- SECTION_SIZE = 2048 on GTX 1050 (1024 threads × 2 elements)
- Reduction tree: index = (threadIdx.x + 1) × 2 × stride - 1 → builds partial sums at positions k×2ⁿ-1
- Reverse tree: same index formula, pushes accumulated sums stride positions to the right
- Convergent thread mapping avoids within-warp divergence until < 32 threads
- **Theoretically work-efficient**, but ~2× the steps of Kogge-Stone (reduction + reverse phases)
- Advantage in limited-resource scenarios (fewer execution units favors work efficiency)
- Convergent thread-to-data mapping: `idx = (threadIdx.x+1)*2*stride - 1` keeps consecutive active threads

### Fig 11.8 — Coarsened Three-Phase Scan

`ch11_coarsened_scan.cu`

Improves work efficiency via thread coarsening (CFACTOR=4, 4096 elements/block):

- **Phase 1:** Coalesced load into shared memory, then each thread sequentially scans its contiguous CFACTOR-element subsection (N - T operations, coalesced writes back)
- **Phase 2:** Block-wide Kogge-Stone scan on the T last-element values (T × log₂(T) operations)
- **Phase 3:** Each thread adds predecessor's accumulated sum to its subsection elements (N - T operations)

**Advantages:**
- Fewer total blocks → less underutilization in final tree iterations
- Work: (N - T) + T·log₂(T) + (N - T) = 2N - 2T + T·log₂(T) — better than pure Kogge-Stone
- Each thread's contiguous subsection enables the O(N) sequential scan to do most of the work
- Section size limited by shared memory, not thread count
- Data-scalable: 1024 threads can handle up to 48K elements (48 KB shared memory)

### Section 11.6 — Three-Kernel Hierarchical Segmented Scan

`ch11_segmented_scan.cu`

Extends segmented scan to arbitrarily large inputs (Fig 11.9). Three kernels:

- **Kernel 1:** Kogge-Stone scan per block + writes last element (block sum) to global S array
- **Kernel 2:** Single-block Kogge-Stone scan on S array (produces global cumulative sums at block boundaries)
- **Kernel 3:** Adds S[blockIdx.x - 1] to every element of each block (filling in cross-block contributions)

**Principle:** Analogous to carry look-ahead in hardware adders. The S array stores scan results at "strategic" positions (block boundaries). Each kernel is a small, fast, focused operation.

## Results (GTX 1050)

| Kernel | Input | Blocks × Threads | Elements/block | Time | Bandwidth | Validation |
|--------|-------|-----------------|----------------|------|-----------|------------|
| Fig 11.3 — Kogge-Stone | 4,096 | 4 × 1024 | 1,024 | 0.008 ms | 4.00 GB/s | PASSED |
| Fig 11.7 — Brent-Kung | 8,192 | 4 × 1024 | 2,048 | 0.011 ms | 5.82 GB/s | PASSED |
| Fig 11.8 — Coarsened (CF=4) | 16,384 | 4 × 1024 | 4,096 | 0.011 ms | 11.64 GB/s | PASSED |
| Section 11.6 — Segmented | 32,768 | 32 × 1024 | 1,024 | 0.066 ms* | 4.00 GB/s | PASSED |

*\*3-kernel total time. All kernels are single-block segmented scans except "Segmented" which produces a full cumulative scan.*

> **Note:** Times are dominated by kernel launch overhead at these input sizes. The coarsened kernel's 11.64 GB/s bandwidth reflects more elements/block reducing per-element overhead.

## Key Concepts

- **Parallel prefix sum (scan):** Each output element i is the sum of all input elements 0..i — seemingly sequential recurrence parallelized via reduction trees
- **Work efficiency:** Kogge-Stone is O(N log N) vs Brent-Kung O(N) — more work done but in fewer steps (log N vs 2 log N steps)
- **Write-after-read hazard:** Unique to scan (not present in reduction). Multiple threads read the same XY location another thread writes in the same iteration → requires temp + double barrier
- **Reduction tree phase (Brent-Kung):** Builds partial sums at rightmost positions (N-1 ops)
- **Reverse tree phase (Brent-Kung):** Distributes accumulated sums to left positions (N-1-log₂(N) ops)
- **Thread coarsening:** CFACTOR determines how many elements each thread processes → fewer blocks → less tree underutilization
- **Hierarchical scan:** Three-kernel approach extends segment-local scan to arbitrary input sizes via global S array accumulation
- **Segmented vs cumulative:** All single-kernel implementations produce segment-local scans. Only the 3-kernel hierarchical version produces the full cumulative scan across the entire input.

## Compilation

```bash
# All kernels target GTX 1050 (sm_61)
nvcc -std=c++17 -arch=sm_61 -O2 -o ch11_kogge_stone_scan ch11_kogge_stone_scan.cu
nvcc -std=c++17 -arch=sm_61 -O2 -o ch11_brent_kung_scan ch11_brent_kung_scan.cu
nvcc -std=c++17 -arch=sm_61 -O2 -o ch11_coarsened_scan ch11_coarsened_scan.cu
nvcc -std=c++17 -arch=sm_61 -O2 -o ch11_segmented_scan ch11_segmented_scan.cu
```

## Reference

Programming Massively Parallel Processors (4th ed.), Kirk, Hwu & El Hajj, 2023
- Fig 11.3 — Kogge-Stone parallel inclusive scan kernel (p. 240)
- Fig 11.4 — Kogge-Stone exclusive scan (p. 243)
- Fig 11.7 — Brent-Kung inclusive scan kernel (p. 250)
- Fig 11.8 — Three-phase coarsened scan (p. 252)
- Fig 11.9 — Hierarchical scan for arbitrary-length inputs (p. 254)
