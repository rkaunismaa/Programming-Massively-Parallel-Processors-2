# Programming Massively Parallel Processors — CUDA Implementation

<img src="https://img.shields.io/badge/CUDA-12.x-76B900?logo=nvidia" alt="CUDA"> <img src="https://img.shields.io/badge/Target-GTX%201050%20(sm__61)-success" alt="GTX 1050"> <img src="https://img.shields.io/badge/Status-21%20chapters%20complete-blue" alt="Chapters">

Hands-on CUDA implementations of every kernel from **Programming Massively Parallel Processors: A Hands-on Approach** (4th Edition, Kirk, Hwu & El Hajj, Morgan Kaufmann 2023).

All code targets a **NVIDIA GeForce GTX 1050** (Pascal, sm_61, 5 SMs, 2 GB VRAM). Also available: RTX 4090 (device 0).

---

## Progress

| Chapter | Topic | Kernels | Status |
|---------|-------|---------|--------|
| 1 | Introduction | — | ✅ README only |
| 2 | Heterogeneous Data Parallel | vec_add (baseline + Fig 2.13), exercises | ✅ |
| 3 | Multidimensional Grids & Data | color_to_grayscale, image_blur, matrix_mul, exercises | ✅ |
| 4 | Compute Architecture & Scheduling | — (architectural concepts) | ✅ README only |
| 5 | Memory Architecture & Data Locality | tiled_matmul (static, dynamic, boundary) | ✅ |
| 6 | Performance Considerations | memory_coalescing_demo, corner_turning, thread_coarsening | ✅ |
| 7 | Convolution | basic, constant_memory, tiled, cached_halo | ✅ |
| 8 | Stencil | basic, tiled, coarsened, register_tiling | ✅ |
| **9** | **Parallel Histogram** | **basic, privatized_global, privatized_shared, coarsened_contiguous, coarsened_interleaved, aggregated** | **✅** |
| 10 | Reduction & Minimizing Divergence | simple, convergent, shared_memory, multiblock, coarsened | ✅ |
| **11** | **Prefix Sum (Scan)** | **Kogge-Stone, Brent-Kung, coarsened, hierarchical segmented** | **✅** |
|| 12 | Merge | basic, tiled, circular_buffer | ✅ |
|| **13** | **Sorting** | **basic_radix_sort, tiled_radix_sort, merge_sort** | **✅** |
|| **14** | **Sparse Matrix Computation** | **spmv_coo, spmv_csr, spmv_ell, hybrid_ell_coo, spmv_jds, coo_to_csr** | **✅** |
|| **15** | **Graph Traversal** | **bfs_push, bfs_pull, bfs_edge, bfs_frontier, bfs_privatized, direction_opt, singleblock** | **✅** |
|| 16 | Deep Learning | conv_forward, unroll, conv_gemm | ✅ |
|| **17** | **Iterative MRI Reconstruction** | **fhd_scatter, fhd_gather, fhd_register, fhd_constant, fhd_struct, fhd_optimized** | **✅** |
||| **18** | **Electrostatic Potential Map** | **dcs_gather, dcs_coarsened, dcs_coalesced** | **✅** |
||| 19 | Parallel Programming & Computational Thinking | — (conceptual) | ✅ README only |
||| **20** | **Heterogeneous Computing Cluster** | **cuda_streams demo, exercises** | **✅** |
||| **21** | **CUDA Dynamic Parallelism** | **bezier_curves, quadtree** | **✅** |
||| 22–23 | (Advanced Practices, Conclusion) | — | ⏳ |

---

## Hardware

The primary testbed is a **GTX 1050** (device 1):
- **Architecture:** Pascal (sm_61)
- **SMs:** 5
- **VRAM:** 2 GB GDDR5 (128-bit bus)
- **Clock:** ~1455 MHz
- **Shared memory/block:** 48 KB
- **Registers/block:** 65,536
- **Max threads/block:** 1024

An RTX 4090 (device 0, sm_89) is also available on the same system for comparison runs.

---

## Model Attribution

Chapters in this project were created using different LLM models:

| Chapter(s) | Model |
|------------|-------|
| 1–7 | Qwen 3.6-27B (via LM Studio, hosted at `https://lmstudio.ai/models/qwen/qwen3.6-27b`) |
|| 8–14 | DeepSeek V4 Flash |
||| 15 | DeepSeek V4 Pro |
||| 16 | DeepSeek V4 Pro |
||| 17 | DeepSeek V4 Flash |
||| 18 | DeepSeek V4 Pro |
||| 19 | DeepSeek V4 Pro |
||| 20 | DeepSeek V4 Pro |
||| 21 | DeepSeek V4 Pro |

This information is tracked in case code style, naming conventions, or behavioural quirks need tracing back to a particular model.

---

## Project Structure

```
PMPP/
├── .hermes/               # Agent working files (plans, etc.)
├── chapters/
│   ├── common/
│   │   └── cuda_utils.cuh # Shared macros, timer, validation helpers
│   ├── ch01_introduction/
│   │   └── README.md
│   ├── ch02_heterogeneous_data_parallel/
│   │   ├── vec_add_baseline.cu
│   │   ├── vec_add_fig_2_13.cu
│   │   └── exercises/
│   ├── ch03_multidimensional_grids_data/
│   │   ├── color_to_grayscale.cu
│   │   ├── image_blur.cu
│   │   ├── matrix_mul.cu
│   │   └── exercises/
│   ├── ch04_compute_architecture_scheduling/
│   │   └── README.md
│   ├── ch05_memory_architecture_data_locality/
│   │   ├── ch05_tiled_matmul_static.cu
│   │   ├── ch05_tiled_matmul_dynamic.cu
│   │   └── ch05_tiled_matmul_boundary.cu
│   ├── ch06_performance_considerations/
│   │   ├── ch06_memory_coalescing_demo.cu
│   │   ├── ch06_corner_turning.cu
│   │   ├── ch06_thread_coarsening.cu
│   │   └── README.md
│   ├── ch07_convolution/
│   │   ├── ch07_basic_convolution.cu
│   │   ├── ch07_constant_memory_convolution.cu
│   │   ├── ch07_tiled_convolution.cu
│   │   ├── ch07_cached_halo_convolution.cu
│   │   └── README.md
│   ├── ch08_stencil/
│   │   ├── ch08_basic_stencil.cu
│   │   ├── ch08_tiled_stencil.cu
│   │   ├── ch08_coarsened_stencil.cu
│   │   ├── ch08_register_tiling_stencil.cu
│   │   └── README.md
│   ├── ch09_histogram/
│   │   ├── ch09_basic_histogram.cu
│   │   ├── ch09_privatized_global.cu
│   │   ├── ch09_privatized_shared.cu
│   │   ├── ch09_coarsened_contiguous.cu
│   │   ├── ch09_coarsened_interleaved.cu
│   │   ├── ch09_aggregated.cu
│   │   └── README.md
│   ├── ch10_reduction/
│   │   ├── ch10_simple_reduction.cu
│   │   ├── ch10_convergent_reduction.cu
│   │   ├── ch10_shared_memory_reduction.cu
│   │   ├── ch10_multiblock_reduction.cu
│   │   ├── ch10_coarsened_reduction.cu
│   │   └── README.md
│   ├── ch11_prefix_scan/
│   │   ├── ch11_kogge_stone_scan.cu
│   │   ├── ch11_brent_kung_scan.cu
│   │   ├── ch11_coarsened_scan.cu
│   │   ├── ch11_segmented_scan.cu
│   │   └── README.md
│   ├── ch13_sorting/
│   │   ├── ch13_basic_radix_sort.cu
│   │   ├── ch13_tiled_radix_sort.cu
│   │   └── ch13_merge_sort.cu
│   ├── ch14_sparse_matrix/
│   │   ├── ch14_spmv_coo.cu
│   │   ├── ch14_spmv_csr.cu
│   │   ├── ch14_spmv_ell.cu
│   │   ├── ch14_spmv_hybrid_ell_coo.cu
│   │   ├── ch14_spmv_jds.cu
│   │   ├── ch14_coo_to_csr.cu
│   │   └── README.md
│   ├── ch15_graph_traversal/
│   │   ├── ch15_bfs_push.cu
│   │   ├── ch15_bfs_pull.cu
│   │   ├── ch15_bfs_edge.cu
│   │   ├── ch15_bfs_frontier.cu
│   │   ├── ch15_bfs_privatized.cu
│   │   ├── ch15_bfs_direction_opt.cu
│   │   ├── ch15_bfs_singleblock.cu
│   │   └── README.md
│   ├── ch16_deep_learning/
│   │   ├── ch16_conv_forward.cu
│   │   ├── ch16_unroll.cu
│   │   ├── ch16_conv_gemm.cu
│   │   └── README.md
│   ├── ch17_iterative_mri/
│   │   ├── ch17_fhd_scatter.cu
│   │   ├── ch17_fhd_gather.cu
│   │   ├── ch17_fhd_register.cu
│   │   ├── ch17_fhd_constant.cu
│   │   ├── ch17_fhd_struct.cu
│   │   ├── ch17_fhd_optimized.cu
│   │   └── README.md
│   ├── ch18_electrostatic/
│   │   ├── ch18_dcs_gather.cu
│   │   ├── ch18_dcs_coarsened.cu
│   │   ├── ch18_dcs_coalesced.cu
│   │   └── README.md
│   ├── ch19_computational_thinking/
│   │   └── README.md
│   ├── ch20_heterogeneous_cluster/
│   │   ├── ch20_cuda_streams.cu
│   │   ├── ch20_exercises.cu
│   │   └── README.md
│   ├── ch21_dynamic_parallelism/
│   │   ├── ch21_bezier_curves.cu
│   │   ├── ch21_quadtree.cu
│   │   ├── ch21_exercises.cu
│   │   └── README.md
│   └── ...               # Future chapters
├── .gitignore
└── README.md              # ← You are here
```

---

## Building & Running

Each `.cu` file compiles independently. There's no top-level build system — just `cd` into a chapter directory and run nvcc:

```bash
cd PMPP/chapters/ch08_stencil
nvcc -std=c++17 -arch=sm_61 -O2 -o ch08_basic_stencil ch08_basic_stencil.cu
./ch08_basic_stencil
```

Common include paths are handled via relative `../common/cuda_utils.cuh`.

### Compilation Flags

| Flag | Purpose |
|------|---------|
| `-std=c++17` | C++17 language standard |
| `-arch=sm_61` | Target GTX 1050 Pascal (sm_61) — **always required** |
| `-O2` | Optimization level 2 (standard) |
| `-o <name>` | Output binary name (matches `.cu` stem) |

> **Note:** All kernels call `cudaSetDevice(1)` to target the GTX 1050. If you only have one GPU, change this to `cudaSetDevice(0)` or remove the call.

---

## Common Utilities (`chapters/common/cuda_utils.cuh`)

All kernels share a common header providing:

- **`CHECK_CUDA(call)`** — Error-checking macro wrapping any CUDA API call
- **`gpu_timer`** — RAII struct wrapping `cudaEvent_t` for kernel timing (`start()`, `stop()`, `elapsed_ms()`)
- **`cpu_allclose(a, b, n, tol)`** — Floating-point validation with relative tolerance
- **`print_device_info(device_id)`** — Prints GPU properties at runtime

---

## Key Results Summary

### Chapters 2–5 (Foundations)

| Kernel | Matrix Size | Time | GFLOPS | Speedup |
|--------|-------------|------|--------|---------|
| Tiled matmul (static, T=16) | 2048×2048 | 50.5 ms | **265.0** | baseline |
| Tiled matmul (boundary check) | 2048×2048 | 50.6 ms | 264.4 | ~1× |

### Chapter 6 (Performance Considerations)

| Kernel | Size | Time | Metric | Speedup |
|--------|------|------|--------|---------|
| Thread coarsening (COARSE=4) | 2048×2048 | 73.10 ms | 235.0 GFLOPS | — |
| Memory coalescing (coalesced) | 1M elements | 0.093 ms | 86.3 GB/s | **9.47×** |
| Memory coalescing (uncoalesced) | 1M elements | 0.878 ms | 9.1 GB/s | baseline |

### Chapter 7 (Convolution)

| Kernel | 1024×1024 | Time | Metric | Improvement |
|--------|-----------|------|--------|-------------|
| Basic | — | 0.98 ms | 107 GB/s eff BW | — |
| Constant-memory filter | — | 0.51 ms | 204.8 GB/s img BW | ~2× |
| Tiled + shared halos | — | 0.27 ms | **193.9 GFLOPS** | ~35× |
| Cached halo (L2) | — | 0.67 ms | 77.9 GFLOPS | — |

### Chapter 8 (Stencil)

| Kernel | 64³ | Time | GFLOPS | OP/B | Key Limitation |
|--------|-----|------|--------|------|----------------|
| Basic | — | 0.049 ms | **63.03** | 0.46 | 7 global loads/thread |
| Tiled (8×8×8) | — | 0.085 ms | 36.45 | 1.37 | 58% halo overhead |
| Coarsened (32×32) | — | 0.089 ms | 34.78 | 2.68 | Overhead on small grid |
| Register tiling | — | **0.077 ms** | **40.34** | — | 4 KB shared mem, +2 regs |

> **Note:** Chapter 8 results use a small 64³ grid where the basic kernel is fastest because the entire dataset fits in L2 cache. Tiling benefits emerge at much larger grid sizes typical in HPC.

### Chapter 9 (Parallel Histogram)

| Kernel | Input | Time | Atomics/sec | Speedup |
|--------|-------|------|-------------|---------|
| Basic atomicAdd | 16M chars, 7 bins | 8.20 ms | 2,047 M | 1.0× |
| Privatized (global mem) | — | 4.84 ms | 3,470 M | 1.7× |
| Privatized (shared mem) | — | 2.12 ms | 7,902 M | 3.9× |
| Coarsened contiguous | — | 1.88 ms | 8,919 M | 4.4× |
| **Coarsened interleaved** | — | **1.09 ms** | **15,357 M** | **7.5×** |
| Aggregated | — | 1.35 ms | 12,390 M | 6.1× |

### Chapter 10 (Parallel Reduction)

| Kernel | Input | Blocks×Threads | Time | Bandwidth |
|--------|-------|----------------|------|-----------|
| Simple (interleaved) | 2,048 | 1 × 1024 | 0.011 ms | 1.45 GB/s |
| Convergent (stride-halving) | 2,048 | 1 × 1024 | 0.009 ms | 1.78 GB/s |
| Shared memory | 2,048 | 1 × 1024 | 0.008 ms | 1.00 GB/s |
| Multiblock | 131,072 | 256 × 256 | 0.024 ms | 22.17 GB/s |
| Coarsened (CF=4) | 262,144 | 128 × 256 | 0.025 ms | **42.45 GB/s** |

### Chapter 11 (Prefix Sum / Scan)

| Kernel | Input | Elements/block | Time | Bandwidth |
|--------|-------|----------------|------|-----------|
| Kogge-Stone (segmented) | 4,096 | 1,024 | 0.008 ms | 4.00 GB/s |
| Brent-Kung (work-efficient) | 8,192 | 2,048 | 0.011 ms | 5.82 GB/s |
| Coarsened (CF=4) | 16,384 | 4,096 | 0.011 ms | **11.64 GB/s** |
| Hierarchical segmented | 32,768 | 1,024 | 0.066 ms* | 4.00 GB/s |

*\\*3-kernel total time. All single-kernel segmented scans produce per-block results; only the hierarchical kernel produces a full cumulative scan.*

### Chapter 15 (Graph Traversal — BFS)

| Kernel | Vertices | Edges | Root | Validation |
|--------|----------|-------|------|:----------:|
| Push (top-down, CSR) | 9 | 15 | 0 | PASS |
| Pull (bottom-up, CSC) | 9 | 15 | 0 | PASS |
| Edge-centric (COO) | 9 | 15 | 0 | PASS |
| Frontier (CSR + atomicCAS) | 9 | 15 | 0 | PASS |
| Privatized (shared mem) | 9 | 15 | 0 | PASS |
| Direction-optimized (Ex2) | 9 | 15 | 0 | PASS |
| Single-block multi-level (Ex3) | 9 | 15 | 0 | PASS |

BFS levels from root 0: `0 1 1 2 2 2 2 2 3`

> **Note:** This is a tiny 9-vertex pedagogical graph. All timings are dominated by kernel launch overhead (~11–20 ms). Meaningful performance data requires larger graphs.

### Chapter 16 (Deep Learning — CNN Inference)

| Kernel | Input | Time | Validation |
|--------|-------|------|:----------:|
| conv_forward (direct) | N=2 C=3 H=18 W=18 K=3 M=4 | — | PASS |
| unroll + GEMM verify | C=3 H=5 W=5 K=2 M=2 | — | PASS |
| conv_gemm (tiled 16×16) | N=2 C=3 H=12 W=12 K=3 M=4 | — | PASS |

### Chapter 17 (Iterative MRI — FHD)

| Kernel | M samples | N voxels | Validation |
|--------|-----------|----------|:----------:|
| fhd_scatter | 512 | 64 | PASS |
| fhd_gather | 512 | 64 | PASS |
| fhd_register | 1024 | 256 | PASS |
| fhd_constant | 2048 | 512 | PASS |
| fhd_struct | 2048 | 512 | PASS |
| fhd_optimized | 8192 | 512 | PASS (tol 5e-3) |

> 52.3× vs CPU on GTX 1050 (fhd_optimized, M=8192, N=512). HW trig tolerance 5e-3.

### Chapter 18 (Electrostatic Potential Map — DCS)

| Kernel | 64×64, 2K atoms | 128×128, 8K atoms | 256×256, 8K atoms |
|--------|:---:|:----:|:----:|
| dcs_gather (Fig 18.6) | 0.47 ms | 6.42 ms | 26.77 ms |
| dcs_coarsened (Fig 18.8) | 0.98 ms | 6.15 ms | 19.66 ms |
| dcs_coalesced (Fig 18.10) | 1.53 ms | 5.98 ms | 19.39 ms |

> Thread coarsening + coalescing yields 27.6% speedup at 256×256. At small grids, basic gather is fastest (coarsening overhead dominates).

### Chapter 14 (Sparse Matrix — SpMV)

| Kernel | Matrix | Nonzeros | Throughput |
|--------|--------|:--------:|:----------:|
| SpMV/COO | 1024×1024, 1% | 10,718 | 905 M nz/s |
| SpMV/CSR | 4096×4096, 0.5% | 83,909 | 2,084 M nz/s |
| SpMV/ELL | 4096×4096, 0.5% | 83,659 | 2,483 M nz/s |

---

## License

Code examples in this repository are provided for educational purposes, accompanying the textbook *Programming Massively Parallel Processors: A Hands-on Approach* (4th Edition).
