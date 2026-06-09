# Chapter 4: Compute Architecture and Scheduling

**Book:** Programming Massively Parallel Processors (Kirk, Hwu & El Hajj, 4th ed.)
**Hardware:** NVIDIA GeForce RTX 4090, sm_89 (Ada Lovelace), 128 SMs, 24 GB VRAM

## Overview
Chapter 4 is a conceptual chapter covering GPU compute architecture and how the
hardware schedules and executes threads. Unlike Chapters 2-3, there are no CUDA
code figures to implement - instead, the exercises test understanding of how
the GPU actually works internally.

## Key Topics Covered
- **4.1** Architecture of a modern GPU (SMs, cores, memory hierarchy)
- **4.2** Block scheduling (arbitrary assignment, transparent scalability)
- **4.3** Synchronization and transparent scalability (__syncthreads())
- **4.4** Warps and SIMD hardware (32 threads per warp, lockstep execution)
- **4.5** Control divergence (thread divergence, SIMD efficiency)
- **4.6** Warp scheduling and latency tolerance (hiding memory latency)
- **4.7** Resource partitioning and occupancy (threads, blocks, registers per SM)
- **4.8** Querying device properties (cudaDeviceProp, cudaOccupancy API)

## Exercises
All 9 exercises solved with detailed worked solutions in `EXERCISES_SOLUTIONS.md`.

### Exercise Summary
| Exercise | Topic | Key Answer |
|----------|-------|------------|
| 1 | Kernel analysis (warps, divergence, SIMD efficiency) | 32 warps, 0 divergence, 100% SIMD efficiency |
| 2 | Vector addition grid size | 2048 threads (4 blocks x 512) |
| 3 | Boundary check divergence | 1 divergent warp |
| 4 | Barrier wait time | 17.1% waiting |
| 5 | __syncthreads() necessity | NO - bad idea, hardware doesn't guarantee warp sync |
| 6 | Maximizing SM threads | 512 threads/block -> 1536 threads |
| 7 | Occupancy calculations | All possible; d and e achieve 100% |
| 8 | Full occupancy analysis | (a) YES, (b) NO (blocks), (c) NO (registers) |
| 9 | Matrix multiply feasibility | NO - exceeds 512 threads/block limit |

## Files
- `EXERCISES_SOLUTIONS.md` - Complete worked solutions for all 9 exercises
- `README.md` - This file
