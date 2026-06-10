# Chapter 12 — Merge

**Book:** *Programming Massively Parallel Processors* (Kirk, Hwu & El Hajj, 4th ed.)  
**Topic:** Parallel ordered merge — dynamically identifying input ranges via the co-rank function.  
**Hardware:** NVIDIA GeForce GTX 1050 (Pascal, sm_61, 5 SMs, 2 GB VRAM)

---

## Kernels Implemented

| File | Figures | Status |
|------|---------|--------|
| `ch12_basic_merge.cu` | Fig 12.2, 12.5, 12.9 | **PASS** — all 18 test cases |
| `ch12_tiled_merge.cu` | Fig 12.11, 12.12, 12.13 | **PASS** — all 18 test cases |
| `ch12_circular_buffer_merge.cu` | Fig 12.16, 12.18–12.20 | **In progress** — single-iteration cases work, multi-iteration has a subtle index wrap bug |

### Basic Merge Kernel

- **`merge_sequential()`** (Fig 12.2) — Standard two-pointer sequential merge
- **`co_rank()`** (Fig 12.5) — Binary search returning number of A elements in first k output positions
- **`merge_basic_kernel()`** (Fig 12.9) — Each thread owns a contiguous output segment, calls `co_rank` on global memory, then `merge_sequential`

**Performance:** 0.06–0.36 GB/s on GTX 1050 (limited by uncoalesced global memory accesses)

### Tiled Merge Kernel

- **Block-level co-rank** (Fig 12.11) — Only thread 0 calls `co_rank` on global memory
- **Cooperative tile loading** (Fig 12.12) — All threads load tiles into `__shared__` coalesced
- **Thread-level co-rank on `__shared__`** (Fig 12.13) — Individual threads call `co_rank` on shared memory, then merge from there

**Performance:** 0.12–3.49 GB/s (10× improvement over basic on balanced inputs). Limitation: ~50% of loaded data is wasted per iteration.

### Circular Buffer Merge Kernel (in progress)

Uses wrap-around circular buffers in shared memory so that unconsumed data persists across iterations. The implementation is complete but has a multi-iteration indexing issue being debugged.

## Key Concepts

### Co-rank Function
`co_rank(k, A, m, B, n)` returns how many elements of the first `k` merged outputs come from array `A`. Uses O(log N) binary search. Enables independent thread work partitioning.

### Dynamic Input Identification
Unlike prior patterns (matmul, convolution, stencil), the input range for each thread depends on data values, not index arithmetic. Makes tiling and coalescing challenging.

### Bug Hunting History
1. **co_rank infinite loop** — delta=0 when search range collapsed. Fix: `if (delta < 1) delta = 1`
2. **Register broadcast** — C_completed not visible to all threads. Fix: shared memory broadcast
3. **Circular index negation** — `ci(-1, ...)` returned -1. Fix: `if (idx < 0) idx += ts`
4. **Subarray vs full-buffer pointers** — `&A_s[ic]` double-offset with circular indexing. Fix: pass full buffer + adjusted start
5. **Buffer occupancy overcount** — Fixed `Aload_init` caused stale data reads when near end of input. Fix: recompute occupancy after refill

## Build & Run

```bash
cd PMPP/chapters/ch12_merge/
nvcc -std=c++17 -arch=sm_61 -O2 -o basic_merge ch12_basic_merge.cu && ./basic_merge
nvcc -std=c++17 -arch=sm_61 -O2 -o tiled_merge ch12_tiled_merge.cu && ./tiled_merge
```

## Test Results

### Basic Merge
```
N=   128 (m=  25,n= 103) | PASS | 0.012 ms | 0.12 GB/s
N=  4096 (m=2048,n=2048) | PASS | 0.135 ms | 0.36 GB/s
N= 32768 (m=16384,n=16384) | PASS | 11.045 ms | 0.14 GB/s (basic)
N=524288 (m=262144,n=262144) | PASS | 66.689 ms | 0.09 GB/s
```

### Tiled Merge
```
N=   128 (m=  64,n=  64) | PASS | 0.013 ms | 0.12 GB/s
N=  4096 (m=2048,n=2048) | PASS | 0.135 ms | 0.36 GB/s
N= 32768 (m=16384,n=16384) | PASS | 0.113 ms | 3.49 GB/s  ← best case
N=524288 (m=262144,n=262144) | PASS | 17.550 ms | 0.36 GB/s
```
