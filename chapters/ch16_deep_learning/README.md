# Chapter 16: Deep Learning

<img src="https://img.shields.io/badge/CUDA-12.x-76B900?logo=nvidia" alt="CUDA"> <img src="https://img.shields.io/badge/Target-GTX%201050%20(sm__61)-success" alt="GTX 1050">

CUDA implementations of convolutional neural network kernels from Chapter 16 of *Programming Massively Parallel Processors: A Hands-on Approach* (4th Edition, Kirk, Hwu & El Hajj, Morgan Kaufmann 2023).

---

## Kernels Implemented

| Kernel | Figure | Description | Status |
|--------|--------|-------------|:------:|
| `ch16_conv_forward` | 16.15 | Direct convolution forward pass — each thread computes one output pixel | ✅ |
| `ch16_unroll` | 16.18 | Unrolls input feature maps into a matrix for GEMM-based convolution | ✅ |
| `ch16_conv_gemm` | 16.16–16.18 | Full GEMM pipeline: unroll + tiled matrix multiply | ✅ |

---

## Data Layout

All arrays use **row-major** order with linearized indexing:

```
X[N][C][H][W]         → X[n*C*H*W + c*H*W + h*W + w]
W_weights[M][C][K][K]  → W[m*C*K*K + c*K*K + p*K + q]
Y[N][M][H_out][W_out] → Y[n*M*H_out*W_out + m*H_out*W_out + h*W_out + w]
```

Where: N = minibatch size, C = input channels, H/W = input spatial dims, K = filter size, M = output feature maps.

---

## Kernel Details

### 1. ch16_conv_forward.cu — Direct Convolution Forward (Fig 16.15)

**Thread organization:**
- 2D thread blocks: `TILE_WIDTH × TILE_WIDTH` (default 16)
- 3D grid: `(M, H_grid × W_grid, N)`
- Each thread computes one output pixel `Y[n][m][h][w]`
- Serial inner loops over channels `c` and filter `(p, q)` accumulate the dot product

**Parallelism:** N × M × H_out × W_out independent threads.

### 2. ch16_unroll.cu — Input Unrolling (Fig 16.18)

**Purpose:** Transform convolution into matrix multiplication by rearranging input data.

**Output:** `X_unroll[(C*K*K)][(H_out*W_out)]` — each column holds all input pixels needed for one output pixel.

**Thread organization:**
- Total threads = C × H_out × W_out
- 1D blocks (BLOCK_SIZE=256)
- Each thread writes K×K elements into one column of X_unroll
- Writes are coalesced: adjacent threads write to adjacent columns

**Verification:** Validates both the unrolled matrix structure and GEMM equivalence (W_filt × X_unroll matches direct convolution).

**Expansion ratio:** up to K²× — each input pixel may be duplicated multiple times due to overlapping convolution windows.

### 3. ch16_conv_gemm.cu — GEMM-Based Convolution (Section 16.4)

**Pipeline (per sample):**
1. **Unroll:** `unroll_kernel` transforms X → X_unroll
2. **Matmul:** `tiled_matmul_kernel` computes Y = W_filt × X_unroll using shared-memory tiling

**Matmul dimensions:**
- A = W_filt: M × (C·K·K)
- B = X_unroll: (C·K·K) × (H_out·W_out)
- C = Y: M × (H_out·W_out)

**Tiling:** TILE_SIZE=16, shared memory tiles for both A and B. Each block computes a 16×16 tile of the output. Boundary conditions handle non-multiples of TILE_SIZE.

---

## Building & Running

```bash
cd PMPP/chapters/ch16_deep_learning

# Direct convolution
nvcc -std=c++17 -arch=sm_61 -O2 -o ch16_conv_forward ch16_conv_forward.cu
./ch16_conv_forward

# Input unrolling
nvcc -std=c++17 -arch=sm_61 -O2 -o ch16_unroll ch16_unroll.cu
./ch16_unroll

# GEMM-based convolution
nvcc -std=c++17 -arch=sm_61 -O2 -o ch16_conv_gemm ch16_conv_gemm.cu
./ch16_conv_gemm
```

---

## Test Results

| Kernel | Test Config | Result |
|--------|-------------|:------:|
| Direct conv | N=2 C=3 H=18 W=18 K=3 M=4 | PASS |
| Unroll + GEMM verify | C=3 H=5 W=5 K=2 M=2 | PASS |
| GEMM pipeline | N=2 C=3 H=12 W=12 K=3 M=4 | PASS |

All kernels validated against CPU reference implementations with relative tolerance 1e-4.

---

## Chapter Structure

- **Section 16.3** — Direct convolution kernel: maps the 4-way parallelism (n, m, h, w) onto a 2D-block/3D-grid CUDA organization.
- **Section 16.4** — GEMM formulation: unrolls input feature maps into an expanded matrix so convolution becomes a single matrix multiply. The tiled matmul uses shared memory to reduce global memory traffic.
- **Section 16.5** — cuDNN overview: discusses NVIDIA's production library which uses lazy on-chip materialization of the unrolled matrix to avoid the memory overhead of the naive GEMM approach.

---

## Exercises (from textbook)

1. Implement the forward pass for the pooling layer (Section 16.2)
2. Analyze layout alternatives: [NCHW] vs [NHWC] vs [CHWN] — memory bandwidth tradeoffs
3. Implement the backward pass for the convolutional layer
4. Analyze read access coalescing in the unroll_Kernel (Fig 16.18)

---

## Key Concepts

- **Data-layout awareness:** Row-major linearization of 4D tensors matters for coalesced access
- **Tiling:** Both the direct conv and GEMM approaches benefit from shared-memory tiling
- **Im2col (unrolling):** Trading increased memory footprint for the computational efficiency of GEMM
- **Expansion ratio:** Approaches K² for large feature maps — can be prohibitive for large filters
- **cuDNN:** Production solution avoids materializing the full unrolled matrix by doing it lazily on-chip
