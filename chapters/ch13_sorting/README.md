# Chapter 13 — Sorting

**Book:** *Programming Massively Parallel Processors* (Kirk, Hwu & El Hajj, 4th ed.)  
**Topic:** Parallel sorting — radix sort (1-bit and 4-bit) and parallel merge sort.  
**Hardware:** NVIDIA GeForce GTX 1050 (Pascal, sm_61, 5 SMs, 2 GB VRAM)

---

## Kernels Implemented

| File | Figures | Status |
|------|---------|--------|
| `ch13_basic_radix_sort.cu` | Fig 13.4, 13.5 | **PASS** — all 8 sizes |
| `ch13_tiled_radix_sort.cu` | Sections 13.4–13.5 | **PASS** — all 8 sizes |
| `ch13_merge_sort.cu` | Section 13.7 | **PASS** — all 8 sizes |

### Basic Radix Sort (ch13_basic_radix_sort.cu)

- **1-bit radix LSD sort**: 16 passes for 16-bit unsigned integers
- Per-block sorting in shared memory using Kogge-Stone inclusive scan
- Two-pointer bucket placement: zeros first, ones after
- Pairwise merge tree using co-rank (Ch12 pattern) for global sort
- **Stable**: scan-based placement preserves thread-index order
- Throughput: ~0.01–0.02 GB/s (limited by 16 global passes + merge overhead)

### Tiled Radix Sort (ch13_tiled_radix_sort.cu)

- **4-bit radix LSD sort**: 4 passes for 16-bit unsigned integers (vs 16 for 1-bit)
- Per-block digit extraction and histogram-based bucket placement
- Position-within-digit computed via O(BLOCK_SIZE) counting per thread
  (deterministic and stable since it iterates in input order)
- Pairwise merge tree for global sort
- **~21% faster** than basic version due to 4× fewer passes

### Parallel Merge Sort (ch13_merge_sort.cu)

- **Comparison-based**: works for any key type with `<` operator
- Bottom-up: odd-even transposition sort on 256-element blocks (base case)
- Iterative pairwise merge using co-rank + sequential merge (Ch12)
- O(N log N) work complexity
- Slower than radix sort on this data (comparison-based overhead)

## Key Concepts

- **Radix sort**: Non-comparison-based, sorts by individual bits (LSD first)
- **Stability**: Equal keys preserve input order — critical for LSD correctness
- **Radix choice**: 4-bit (16 buckets) reduces passes by 4× vs 1-bit
- **Scan-based placement**: Kogge-Stone scan on bit flags gives bucket positions
- **Merge sort**: Comparison-based, uses Ch12 co-rank for parallel merge
- **Odd-even sort**: Simple comparison network for small per-block base case

## Performance Results (GTX 1050, 1M elements)

| Kernel | n=1M | Throughput |
|--------|------|------------|
| Basic radix sort (1-bit) | 490 ms | 0.01 GB/s |
| Tiled radix sort (4-bit) | 386 ms | 0.02 GB/s |
| Parallel merge sort | 808 ms | 0.01 GB/s |

Radix sort outperforms merge sort due to simpler per-element work (bit extraction
vs comparison + co-rank). The tiled 4-bit version is fastest with only 4 passes.

## Build & Run

```bash
cd PMPP/chapters/ch13_sorting/
nvcc -std=c++17 -arch=sm_61 -O2 -o basic_radix_sort ch13_basic_radix_sort.cu && ./basic_radix_sort
nvcc -std=c++17 -arch=sm_61 -O2 -o tiled_radix_sort ch13_tiled_radix_sort.cu && ./tiled_radix_sort
nvcc -std=c++17 -arch=sm_61 -O2 -o merge_sort ch13_merge_sort.cu && ./merge_sort
```
