# Chapter 20 — Programming a Heterogeneous Computing Cluster

**Type**: CUDA + MPI (conceptual implementation due to no MPI on this system)
**Hardware**: GTX 1050, sm_61 (Pascal), 5 SMs, 2 GB VRAM

## Overview

This chapter introduces joint CUDA/MPI programming for HPC clusters with
heterogeneous computing nodes. Key CUDA concepts: streams, pinned memory,
and asynchronous data transfers to overlap computation with communication.

## Contents

| File | Description | Status |
|------|-------------|--------|
| `ch20_cuda_streams.cu` | Streams + pinned memory + async stencil demo | ✅ PASS |
| `ch20_exercises.cu` | Exercise 1-4 solutions | ✅ |

## CUDA Streams Demo

Demonstrates three patterns from Ch20 without requiring MPI:

| Test | Pattern | Time (64³) |
|------|---------|------------|
| Single stream | Baseline — one kernel covers entire grid | 0.068 ms |
| Dual streams | Two concurrent kernels on grid halves | ~0.07 ms |
| Async overlap | Kernel in stream1 + D2H copy in stream0 overlapped | ~0.23 ms |

> The dual-stream kernel timing is ~0.068ms, consistent with Ch8's basic stencil
> (0.049ms for 64³). Stream overlap doesn't show dramatic speedup on GTX 1050
> due to its limited 5 SMs — benefits are more pronounced on multi-GPU clusters.

## Key CUDA Concepts Demonstrated

1. **Pinned memory** (`cudaHostAlloc`): Page-locked host memory for DMA access.
   Avoids the two-step copy (user buffer → pinned buffer → device) that standard
   `malloc` requires. Essential for async copies and MPI buffer staging.

2. **CUDA streams** (`cudaStream_t`): Ordered sequences of operations. Operations
   in different streams can execute concurrently. Enables:
   - Multiple kernels running simultaneously (if SMs available)
   - Memory copies overlapping with kernel execution

3. **Async memory copies** (`cudaMemcpyAsync`): Non-blocking transfers that
   can overlap with kernel execution in other streams.

4. **Two-stage overlap** (Fig 20.12): Stage 1 computes boundary data for neighbors;
   Stage 2 overlaps MPI communication with interior computation.

## Exercise Solutions

### Exercise 1: Grid Partitioning (64×64×2048, 16 compute processes, 25-pt stencil)

| Question | Internal Process | Edge Process |
|----------|:---:|:---:|
| (a) Output points | 524,288 | 524,288 |
| (b) Halo points needed | 16,384 (2 sides × 2×64×64) | 8,192 (1 side) |
| (c) Stage 1 boundary points | 32,768 (2 sides × 4×64×64) | 16,384 (1 side) |
| (d) Stage 2 internal points | 491,520 (120×64×64) | 491,520 |
| (e) Stage 2 bytes sent | 65,536 (2 sides × 2×64×64×4B) | 32,768 (1 side) |

### Exercise 2: MPI Element Size
Answer: **(c) 4 bytes** — MPI_FLOAT = 4 bytes. 1000 elements × 4B = 4000 bytes.

### Exercise 3: MPI True/False
Answer: **(b) only** — `MPI_Recv()` is blocking by default.
- (a) False: `MPI_Send()` implementation-dependent
- (c) False: messages can be any size
- (d) False: separate address spaces

### Exercise 4: CUDA-Aware MPI

With CUDA-aware MPI (MVAPICH2, OpenMPI with CUDA support), device pointers
can be passed directly to `MPI_Send`/`MPI_Recv`, eliminating:

```c
// REMOVED: Pinned bounce buffers
cudaHostAlloc(&h_left_halo, ...);

// REMOVED: D2H copy before MPI_Sendrecv
cudaMemcpyAsync(h_left_boundary, d_output + offset, ...);

// REPLACED: MPI_Sendrecv uses device pointers directly
MPI_Sendrecv(d_output + num_halo_points, ..., d_output + right_halo_offset, ...);

// REMOVED: H2D copy after MPI_Sendrecv
cudaMemcpyAsync(d_output + left_halo_offset, h_left_halo, ...);
```

## MPI Prerequisites

The full distributed stencil requires MPI (not installed on this system):
```bash
# Install MPI:
sudo apt install openmpi-bin libopenmpi-dev

# Compile MPI+CUDA:
mpicxx -std=c++17 -o ch20_mpi_stencil ch20_mpi_stencil.cu \
  -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcudart

# Run with 17 processes (1 server + 16 compute):
mpirun -np 17 ./ch20_mpi_stencil
```

## Key Architecture Pattern

```
┌─────────────────────────────────────────────────┐
│                   MPI_COMM_WORLD                 │
│  ┌──────────┐  ┌──────┐  ┌──────┐     ┌──────┐ │
│  │  Server   │  │ Proc │  │ Proc │ ... │ Proc │ │
│  │  (rank 0) │  │  1   │  │  2   │     │  16  │ │
│  │           │  │ GPU  │  │ GPU  │     │ GPU  │ │
│  └──────────┘  └──────┘  └──────┘     └──────┘ │
│       ▲            ↕          ↕            ↕     │
│       │     MPI_Sendrecv  (halo exchange)        │
│       └────────── MPI_Gather ──────────────────┘ │
└─────────────────────────────────────────────────┘
```

## References

- Gropp, W., Lusk, E., Skjellum, A. (1999). *Using MPI*, 2nd Ed. MIT Press.
- Rodrigues, C.I. et al. (2008). GPU acceleration of cutoff pair potentials.
  *Proceedings of the Fifth Conference on Computing Frontiers*.
