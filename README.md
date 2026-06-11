# Programming Massively Parallel Processors — CUDA Implementation

<img src="https://img.shields.io/badge/CUDA-12.x-76B900?logo=nvidia" alt="CUDA"> <img src="https://img.shields.io/badge/Target-GTX%201050%20(sm__61)-success" alt="GTX 1050"> <img src="https://img.shields.io/badge/Status-13%20chapters%20complete-blue" alt="Chapters">

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
|| 14–19 | (Advanced topics) | — | ⏳ |

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
| 8–13 | DeepSeek V4 Flash |

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

*\*3-kernel total time. All single-kernel segmented scans produce per-block results; only the hierarchical kernel produces a full cumulative scan.*

---

## License

Code examples in this repository are provided for educational purposes, accompanying the textbook *Programming Massively Parallel Processors: A Hands-on Approach* (4th Edition).
