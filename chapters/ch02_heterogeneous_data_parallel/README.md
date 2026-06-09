# Chapter 2: Heterogeneous Data Parallel Computing

## Summary
Chapter 2 introduces CUDA C programming through the lens of vector addition — the "Hello World" of GPU computing. It covers the fundamental concepts of data parallelism, device memory management, kernel functions, and the host-device execution model. The chapter establishes the basic pattern of allocating device memory, transferring data, launching kernels, and retrieving results.

## Concepts Covered
- Data parallelism vs task parallelism
- CUDA C program structure: host code + device code
- Device global memory allocation (`cudaMalloc`, `cudaFree`)
- Host-device data transfer (`cudaMemcpy`)
- Kernel functions with `__global__` qualifier
- Thread hierarchy: grid → blocks → threads
- Built-in variables: `threadIdx`, `blockIdx`, `blockDim`
- Kernel launch configuration: `<<<grid_size, block_size>>>`
- Index mapping: `i = blockIdx.x * blockDim.x + threadIdx.x`
- NVCC compilation process (host code + PTX device code)
- Function qualifiers: `__global__`, `__device__`, `__host__`

## Files Generated
| File | Book Reference | Concept |
|------|---------------|---------|
| ch02_vec_add_fig_2_13.cu | Figure 2.10 + Figure 2.13 | Complete vector addition with kernel and host wrapper |
| ch02_vec_add_baseline.cu | Figure 2.4 | Sequential CPU baseline for comparison |
| exercises/ex01.cu | Exercise 2.1 | Thread-to-data index mapping verification |
| exercises/ex02.cu | Exercise 2.2 | Each thread processes two adjacent elements |
| exercises/ex03.cu | Exercise 2.3 | Two-section processing pattern |

## Figures Skipped (diagrams/illustrations — no compilable code)
| Figure | Description |
|--------|-------------|
| Figure 2.1 | Color image to grayscale conversion illustration |
| Figure 2.2 | Data parallelism in image-to-grayscale (conceptual diagram) |
| Figure 2.3 | CUDA program execution flow (host/device timeline) |
| Figure 2.5 | Outline of revised vecAdd function (skeleton/pseudocode) |
| Figure 2.6 | cudaMalloc/cudaFree API function signatures |
| Figure 2.7 | cudaMemcpy API function signature |
| Figure 2.8 | More complete vecAdd skeleton (Part 1 and 3 filled) |
| Figure 2.9 | Thread block/thread hierarchy diagram (256 threads per block) |
| Figure 2.11 | CUDA C keywords table (__global__, __device__, __host__) |
| Figure 2.12 | Kernel call statement with execution configuration |
| Figure 2.14 | NVCC compilation process overview diagram |

## Key Takeaways
- CUDA extends C with minimal new syntax for heterogeneous computing
- The grid of threads replaces the sequential for-loop — each thread = one loop iteration
- Device memory is separate from host memory; explicit transfers are required
- Block size should be a multiple of warp size (32) for hardware efficiency
- The `if (i < n)` guard handles cases where grid size exceeds data size
- CUDA programs compile to both host object code and PTX (device virtual machine code)

## Build
```bash
cd /home/rob/Data2/Programming\ Massively\ Parallel\ Processors\ Hermes/PMPP
mkdir -p build && cd build
cmake ..
make ch02_vec_add_fig_2_13 ch02_vec_add_baseline
# Or compile individually:
nvcc -std=c++17 -arch=sm_61 -O2 ../chapters/ch02_heterogeneous_data_parallel/ch02_vec_add_fig_2_13.cu -o vec_add
```
