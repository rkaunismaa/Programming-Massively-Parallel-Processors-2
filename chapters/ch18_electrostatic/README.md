# Chapter 18 — Electrostatic Potential Map

**Hardware**: GTX 1050, sm_61 (Pascal), 5 SMs, 2 GB VRAM
**Compiler**: nvcc -std=c++17 -arch=sm_61 -O2

## Overview

Direct Coulomb Summation (DCS) for electrostatic potential maps, from VMD
(Visual Molecular Dynamics). Each grid point's potential = sum of contributions
from all atoms: energy_j = Σ(charge_i / distance_ij).

## Kernels Implemented

| Kernel | File | Figure | Description |
|--------|------|--------|-------------|
| Gather | `ch18_dcs_gather.cu` | 18.6 | One thread per grid point, atom data in __constant__ memory |
| Coarsened | `ch18_dcs_coarsened.cu` | 18.8 | COARSEN_FACTOR=4, consecutive grid points per thread, reuse dy/dz/charge |
| Coalesced | `ch18_dcs_coalesced.cu` | 18.10 | Coarsening + interleaved assignment for coalesced global writes |

## Validation Results

All kernels validated PASS on GTX 1050. Atom data chunked to fit in 64KB constant memory (CHUNK_SIZE=4096 atoms/chunk = 16KB = 16,384 floats).

### 64×64 grid, 2,000 atoms
| Kernel | Time |
|--------|------|
| Gather | 0.471 ms |
| Coarsened | 0.975 ms |
| Coalesced | 1.527 ms |

### 128×128 grid, 8,000 atoms
| Kernel | Time | vs Gather |
|--------|------|-----------|
| Gather | 6.416 ms | — |
| Coarsened | 6.153 ms | 4.2% faster |
| Coalesced | 5.976 ms | 6.9% faster |

### 256×256 grid, 8,000 atoms
| Kernel | Time | vs Gather |
|--------|------|-----------|
| Gather | 26.768 ms | — |
| Coarsened | 19.663 ms | 26.5% faster |
| Coalesced | 19.390 ms | 27.6% faster |

## Performance Analysis

- **Coarsening** reduces constant memory accesses from 16 to 4 per 4 grid points and FLOPs from 48 to 24 — substantial improvement at larger grid sizes.
- **Coalescing** provides marginal additional gain (~1%) by interleaving grid point assignment so adjacent threads write adjacent memory locations.
- At small grids (64×64), coarsening overhead dominates; the basic gather is fastest.

## Key Concepts

1. **Scatter vs Gather**: Atom-centric parallelization (scatter) requires atomics; grid-centric (gather) uses owner-computes, no atomics needed.
2. **Constant memory caching**: Atom data in `__constant__` memory — broadcast to all threads in a warp, effectively free for repeated reads.
3. **Thread coarsening**: Each thread processes multiple grid points, reusing dy (y-distance), dz (z-distance), dy²+dz², and charge in registers.
4. **Memory coalescing**: Interleaved grid point assignment (stride = blockDim.x) ensures adjacent threads write adjacent locations.
5. **Cutoff binning** (Section 18.5, conceptual): For large volumes, each grid point only considers atoms within a cutoff radius, reducing O(n²) to O(n) complexity via spatial binning.

## Bug Fix: Fig 18.8 Index Formula

The book's Fig 18.8 uses `i = blockIdx.x * blockDim.x * COARSEN_FACTOR + threadIdx.x`
which causes overlapping grid point assignments between adjacent threads
(e.g., thread 0 handles [0,3], thread 1 handles [1,4]). Corrected to:
`i = blockIdx.x * blockDim.x * COARSEN_FACTOR + threadIdx.x * COARSEN_FACTOR`
(e.g., thread 0 handles [0,3], thread 1 handles [4,7]).
