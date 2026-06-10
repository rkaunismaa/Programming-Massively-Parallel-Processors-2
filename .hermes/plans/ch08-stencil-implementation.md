# Chapter 8: Stencil Implementation Plan

> **For Hermes:** Implement kernels one-by-one, verify each compiles and passes validation on GTX 1050, then commit.

**Goal:** Implement four 3D seven-point stencil sweep kernels from Chapter 8, validate on GTX 1050 (sm_61, device 1).

**Architecture:** 3D structured grid, seven-point stencil (center + x, y, z neighbors), boundary ghost cells preserved. Grid initialized with a known function (e.g., sine wave in 3D) and validated against CPU reference.

**Tech Stack:** CUDA C++, nvcc, GTX 1050 (sm_61)

**Progress:**
- [x] Task 1: ch08_basic_stencil.cu — Fig 8.6 naive parallel stencil
- [x] Task 2: ch08_tiled_stencil.cu — Fig 8.8 shared memory tiling
- [x] Task 3: ch08_coarsened_stencil.cu — Fig 8.10 thread coarsening (z-direction)
- [x] Task 4: ch08_register_tiling_stencil.cu — Fig 8.12 register tiling + coarsening
- [x] Task 5: README.md with results table
- [x] Task 6: Commit
