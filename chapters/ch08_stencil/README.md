# Chapter 8: Stencil (Figures 8.6, 8.8, 8.10, 8.12)

**GTX 1050** | sm_61 Pascal, 5 SMs, 1455 MHz, ~2 GB VRAM

## Overview

Stencil computations are foundational to finite-difference methods for solving partial differential equations in fluid dynamics, heat transfer, weather forecasting, and electromagnetics. This chapter implements a 3D seven-point Laplacian stencil sweep with four increasingly sophisticated kernels.

All kernels compute `∇²f` using the finite-difference approximation:
```
∇²f(i,j,k) ≈ -6·f(i,j,k) + f(i-1,j,k) + f(i+1,j,k) + f(i,j-1,k) + f(i,j+1,k) + f(i,j,k-1) + f(i,j,k+1)
```

## Kernels Implemented

| # | File | Figure | Concept | Block | Shared Mem | OP/B | Time (ms) | GFLOPS | Status |
|---|------|--------|---------|-------|-----------|------|-----------|--------|--------|
| 1 | `ch08_basic_stencil` | Fig 8.6 | Naive parallel 3D stencil — each thread loads 7 inputs independently from global memory | 8×8×8 | 0 B | 0.46 | 0.049 | 63.03 | ✅ PASSED |
| 2 | `ch08_tiled_stencil` | Fig 8.8 | Shared memory tiling — 8×8×8 input tile (512 threads), 6×6×6 output tile | 8×8×8 | 2 KB | 1.37 | 0.085 | 36.45 | ✅ PASSED |
| 3 | `ch08_coarsened_stencil` | Fig 8.10 | Thread coarsening in z — 32×32 2D block iterates over z-planes, 3 planes in shared mem | 32×32 | 13 KB | 2.68 | 0.089 | 34.78 | ✅ PASSED |
| 4 | `ch08_register_tiling_stencil` | Fig 8.12 | Register tiling of z-neighbors — inPrev/inNext in registers, only 1 plane in shared mem | 32×32 | 4 KB | — | 0.077 | 40.34 | ✅ PASSED |

## Key Concepts

### Basic Stencil (Fig 8.6)
- Each thread computes one output grid point, loading 7 input values from global memory
- 13 floating-point operations (7 multiplies + 6 additions)
- Arithmetic intensity: 0.46 OP/B — extremely low
- Boundary ghost cells are preserved (not updated)

### Shared Memory Tiling (Fig 8.8)
- Input tile is BLOCK_DIM³ = 8×8×8 (512 threads, max block size)
- Output tile is (BLOCK_DIM-2)³ = 6×6×6 (216 points)
- 58% of input tile elements are *halo cells* (loaded but not computed)
- **Result: slower than basic kernel** — the small 3D tile size creates too much overhead:
  - Poor coalescing: warp covering 4 different rows in the 8×8×8 tile
  - 58% halo overhead dramatically reduces reuse (vs ~12% for 2D convolution)
  - Theoretical upper bound: 3.25 OP/B, actual at T=8: only 1.37 OP/B

This motivates thread coarsening.

### Thread Coarsening (Fig 8.10)
- **Key insight**: use a 2D thread block (T×T) instead of 3D (T×T×T)
- Each thread processes a *column* of output grid points in the z-direction
- Only 3 z-planes of shared memory needed at any time → T can be 32 (1024 threads)
- Shared memory: 3 × 34×34 × 4 B = 13 KB
- Theoretical OP/B: 2.68 (at T=32), much closer to the 3.25 upper bound
- Better coalescing: contiguous column access per warp

### Register Tiling (Fig 8.12)
- **Key insight**: for a 7-point stencil, z-neighbors are each accessed by exactly ONE thread
- inPrev and inNext move to registers (z-1 and z+1 per thread)
- Only inCurr_s (current z-plane) stays in shared memory for x-y neighbor sharing
- Shared memory reduced by 2/3: 1 × 34×34 × 4 B = 4 KB
- Trade-off: +2 registers per thread (inPrev, inNext)
- Only slightly faster on GTX 1050 with 64³ grid due to tiny dataset

## Performance Notes (64³ grid)

On a small 64³ grid (0.5 MB), the **basic kernel** is the fastest because:
1. The entire grid fits in L2 cache (~512 KB on GTX 1050)
2. No shared memory management overhead or synchronization barriers
3. Simple indexing with no cooperative loading loops

The tiled/coarsened/register variants are designed for **much larger grids** (e.g., 256³ or 512³ on real HPC workloads) where the basic kernel becomes severely bandwidth-bound and the tiling overhead is amortized over many more computations.

## Compile Commands

```bash
cd PMPP/chapters/ch08_stencil
nvcc -std=c++17 -arch=sm_61 -O2 -o ch08_basic_stencil ch08_basic_stencil.cu
nvcc -std=c++17 -arch=sm_61 -O2 -o ch08_tiled_stencil ch08_tiled_stencil.cu
nvcc -std=c++17 -arch=sm_61 -O2 -o ch08_coarsened_stencil ch08_coarsened_stencil.cu
nvcc -std=c++17 -arch=sm_61 -O2 -o ch08_register_tiling_stencil ch08_register_tiling_stencil.cu
```

## Comparison with Convolution (Chapter 7)

| Aspect | Convolution (Ch 7) | Stencil (Ch 8) |
|--------|-------------------|----------------|
| Typical dimensions | 2D images | 3D volumetric grids |
| Filter/size | Small, fixed (3×3, 5×5) | Order-1 spatial: 7 points |
| Corner participation | Yes (all neighbors) | No (axial neighbors only) |
| Tile size | 32×32 feasible | T=8 max in 3D (512 threads) |
| Halo overhead | ~12% (32×32 tile) | ~58% (8×8×8 tile) |
| Key optimization | Shared memory tiling | Thread coarsening + register tiling |
| Arithmetic intensity | 4.5 OP/B (3×3) | 3.25 OP/B upper bound |

## Reference

All kernels validated against CPU reference (`cpu_stencil_sweep`) using `cpu_allclose`. Grid: 64³ float array initialized with `sin(i*0.1) * cos(j*0.1) * sin(k*0.1)`. Stencil: 3D seven-point Laplacian (coefficients: -6 for center, +1 for each of 6 neighbors).
