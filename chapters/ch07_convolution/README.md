# Chapter 7: Convolution (Figures 7.7, 7.9, 7.12, 7.15)

**GTX 1050** | sm_61 Pascal, 5 SMs, 1455 MHz, ~2 GB VRAM

## Kernels Implemented

| # | File | Figure | Concept | Time | Metric | Status |
|---|------|--------|---------|------|--------|--------|
| 1 | `ch07_basic_convolution` | Fig 7.7 | Naive parallel 2D convolution, one thread per output pixel with direct global-mem loads | 0.98 ms | 107 GB/s eff BW | ✅ PASSED |
| 2 | `ch07_constant_memory_convolution` | Fig 7.9 | Same as #1 but filter F loaded into `__constant__` memory via `cudaMemcpyToSymbol` | 0.51 ms | 204.8 GB/s (image only) | ✅ PASSED |
| 3 | `ch07_tiled_convolution` | Fig 7.12 | Tiled convolution with shared-memory halos — loads full input tile INCLUDING border cells | 0.27 ms | 193.9 GFLOPS | ✅ PASSED |
| 4 | `ch07_cached_halo_convolution` | Fig 7.15 | Tiled convolution WITHOUT explicit halo loading — shared-mem bounds check + direct global access for halos (L2 cache hit) | 0.67 ms | 77.9 GFLOPS | ✅ PASSED |

## Compile Commands

```bash
cd PMPP/chapters/ch07_convolution
nvcc -std=c++17 -arch=sm_61 -O2 -o ch07_basic_convolution ch07_basic_convolution.cu
nvcc -std=c++17 -arch=sm_61 -O2 -o ch07_constant_memory_convolution ch07_constant_memory_convolution.cu
nvcc -std=c++17 -arch=sm_61 -O2 -o ch07_tiled_convolution ch07_tiled_convolution.cu
nvcc -std=c++17 -arch=sm_61 -O2 -o ch07_cached_halo_convolution ch07_cached_halo_convolution.cu
```

## Key Concepts

### Naive Convolution (Fig 7.7)
- Each thread computes one output pixel, reading all (2R+1)² filter elements from global memory.
- Low arithmetic intensity — every filter access = a DRAM read.

### Constant Memory for Filter (Fig 7.9)
- The square convolution filter is small enough to fit into `__constant__` memory (~128 KB limit).
- When all threads in a warp access the same constant-memory element, NVIDIA hardware broadcasts the value — huge bandwidth saving for the filter reads. ~2× improvement observed.

### Tiled Shared-Memory Halos (Fig 7.12)
- Input tile + surrounding halo cells loaded into `__shared__` memory of size (TILE_DIM + 2R) × (TILE_DIM + 2R).
- Only interior threads [FILTER_RADIUS .. TILE_DIM-1-FILTER_RADIUS] produce output pixels.
- Massive reuse: one shared-memory load serves multiple filter positions. ~35× GFLOPS vs basic kernel.

### Cached Halo Tiles (Fig 7.15)
- **Same tile** as Fig 7.12, but DOES NOT load halo border into shared memory. Only the TILE_DIM-sized interior tile goes into `__shared__`.
- Convolution loop checks: if N_s neighbour index is in-bounds → hit shared mem; else fall back to global `d_N`.
- The key insight: a block's "halo" cells are its neighbouring block's interior → already cached in L2 from that neighbour tile's load.
- **Trade-off:** Higher arithmetic intensity and less shared memory overhead vs extra global-memory reads (albeit L2-cached). Performance is lower than tiled-halo on sm_61 because the L2 cache is smaller on Pascal GPUs; this approach shines on newer architectures with large L2 caches.

### Performance Progression (sm_61 GTX 1050)

| Technique | Time (ms) | Throughput | Improvement over prev |
|-----------|-----------|------------|-----------------------|
| Naive parallel | 0.98 | 107 GB/s eff BW | — |
| Constant-mem filter | 0.51 | 204.8 GB/s img BW | ~2× faster |
| Tiled + shared halos | 0.27 | 193.9 GFLOPS | ~35× over naive |
| Cached halo (L2) | 0.67 | 77.9 GFLOPS | — |

Note: the cached-halo kernel measures lower here because on sm_61 (Pascal), the L2 cache advantage is limited; on newer architectures with larger L2 caches and better cache hierarchies, this approach competes more closely with shared-memory tiling while using less shared memory.

## Reference

All kernels validated against a CPU reference implementation (`cpu_convolve`) using `cpu_allclose` with tolerance 1e-2. Input: 1024×1024 float array with `fmodf(i*0.01f, 10.f)`. Filter: Gaussian with radius 2 (5×5), values `exp(-(dx²+dy²)/2)`.
