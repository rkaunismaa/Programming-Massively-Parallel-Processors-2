# Chapter 17: Iterative MRI Reconstruction

Hardware: GTX 1050 (Pascal sm_61, 5 SMs, ~2 GB VRAM)

## Overview

Implements the FHD (F^H · D) computation from iterative MRI reconstruction.
The computation sums contributions from M k-space samples to N image-space
voxels using complex exponential evaluation (sin/cos per sample-voxel pair).

## Optimization Progression

| Kernel | Description | Key Technique | OP/B Ratio |
|--------|-------------|---------------|:----------:|
| `ch17_fhd_scatter` (Fig 17.5+17.7) | Scatter approach — one thread per k-space sample, scatters to all voxels with atomics | Baseline scatter | ~0.23 |
| `ch17_fhd_gather` (Fig 17.10+17.7) | Gather approach — loop interchange, one thread per voxel, no atomics | Loop interchange | ~0.23 |
| `ch17_fhd_register` (Fig 17.11+17.7) | Register promotion for x[n],y[n],z[n],rFhD[n],iFhD[n] | Registers | ~0.46 |
| `ch17_fhd_constant` (Fig 17.13+17.12) | K-space data in constant memory, chunked for 64KB limit | Constant cache | ~1.63 |
| `ch17_fhd_struct` (Fig 17.16+17.15) | Array-of-structs k-space layout — all 3 coords in one cache line | Struct AoS | ~1.63 |
| `ch17_fhd_optimized` (Fig 17.17) | Hardware trig (__sinf/__cosf via SFU) + all above | HW SFU trig | ~3.25 |

## Validation Results

| Kernel | M | N | GPU Time | vs CPU | Status |
|--------|:--:|:--:|----------|:------:|:------:|
| scatter | 512 | 64 | 0.080 ms | — | PASS |
| gather | 512 | 64 | 0.184 ms | — | PASS |
| register | 512 | 64 | 0.180 ms | — | PASS |
| constant | 2048 | 64 | 0.610 ms | — | PASS |
| struct | 2048 | 64 | 0.610 ms | — | PASS |
| optimized | 8192 | 512 | 0.660 ms | 52.3× | PASS* |

\* Hardware trig uses relaxed tolerance (5e-3) — acceptable for clinical MRI
  (PSNR degrades only 0.1 dB: 27.6 → 27.5 dB per the textbook).

## Key Concepts

- **Scatter vs Gather**: Scatter (one thread per input) requires atomics;
  gather (one thread per output) avoids atomics but needs loop interchange
- **Loop Fission**: Splitting Mu computation from FHD accumulation enables
  independent parallelization of each step
- **Loop Interchange**: Swapping inner/outer loops to make the output loop
  parallelizable (n-loop becomes outer → one thread per voxel)
- **Register Promotion**: Moving x[n], y[n], z[n], rFhD[n], iFhD[n] to
  registers reduces 14 global accesses to 7 per iteration
- **Constant Memory**: k-space data is read-only and shared across all threads
  — ideal for constant cache with warp broadcast
- **Chunking**: Constant memory limited to 64KB; large k-space datasets
  are processed in chunks with multiple kernel launches
- **Struct Layout (AoS)**: Grouping kx[m], ky[m], kz[m] into a struct ensures
  all 3 components occupy one cache line, reducing cache pressure
- **Hardware Trig**: `__sinf()` / `__cosf()` via SFU — 0.1 dB PSNR loss

## Build & Run

```bash
cd chapters/ch17_iterative_mri
nvcc -std=c++17 -arch=sm_61 -O2 -o ch17_fhd_optimized ch17_fhd_optimized.cu
./ch17_fhd_optimized
```
