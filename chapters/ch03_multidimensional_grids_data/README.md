# Chapter 3: Multidimensional Grids and Data

**Book:** Programming Massively Parallel Processors (Kirk, Hwu & El Hajj, 4th ed.)
**Hardware:** NVIDIA GeForce RTX 4090, sm_89 (Ada Lovelace), 128 SMs, 24 GB VRAM

## Overview
Chapter 3 covers how threads are organized in multidimensional grids and blocks,
and how to map those thread coordinates to multidimensional data (images, matrices).
Key concepts: 2D grid/block configuration, row-major linearization, boundary handling.

## Code Examples

### Figure 3.4 — Color-to-Grayscale Conversion
- **File:** `ch03_color_to_grayscale_fig_3_4.cu`
- **Binary:** `color_to_grayscale`
- **Concept:** 2D thread grid mapping to image pixels; each thread converts one RGB pixel
  to grayscale using luminance formula (0.21*R + 0.72*G + 0.07*B)
- **Result:** 1920x1080 image, PASS, 10.048 ms

### Figure 3.8 — Image Blur Kernel
- **File:** `ch03_image_blur_fig_3_8.cu`
- **Binary:** `image_blur`
- **Concept:** Nested loops over a (2*BLUR_SIZE+1)^2 patch; each thread averages
  neighboring pixels; boundary handling via conditional checks inside the loop
- **Result:** 1920x1080 image, 3x3 patch, PASS, 3.258 ms

### Figure 3.11 — Matrix Multiplication
- **File:** `ch03_matrix_mul_fig_3_11.cu`
- **Binary:** `matrix_mul`
- **Concept:** Each thread computes one P[row][col] element as dot product of
  M's row and N's column; square matrices, row-major linearization
- **Result:** 1024x1024 matrices, PASS, 5.859 ms, 366.53 GFLOPS

## Exercises

### Exercise 3.1a — One Thread Per Output Row
- **File:** `exercises/ex01a_matmul_one_row.cu`
- **Binary:** `exercises/ex01a`
- **Concept:** Each thread computes an entire row of P; grid has Width blocks in
  y-dimension; inner loops over columns and k dimension
- **Result:** 1024x1024, PASS, 118.055 ms, 18.19 GFLOPS
- **Note:** Low GFLOPS — each thread does Width^2 work, poor parallelism

### Exercise 3.1b — One Thread Per Output Column
- **File:** `exercises/ex01b_matmul_one_col.cu`
- **Binary:** `exercises/ex01b`
- **Concept:** Each thread computes an entire column of P; grid has Width blocks
  in x-dimension; inner loops over rows and k dimension
- **Result:** 1024x1024, PASS, 36.971 ms, 58.09 GFLOPS
- **Note:** Better than 3.1a due to memory access patterns (column writes are strided)

### Exercise 3.2 — Matrix-Vector Multiplication
- **File:** `exercises/ex02_matvec.cu`
- **Binary:** `exercises/ex02_matvec`
- **Concept:** A = B * C where B is square matrix, C is vector; each thread computes
  one output vector element as dot product of one matrix row with the vector
- **Result:** 4096x4096 matrix, PASS, 12.101 ms, 2.77 GFLOPS

## Compilation
```bash
nvcc -std=c++17 -arch=sm_89 -O2 <source>.cu -o <binary>
```

## Performance Comparison (Exercise 3.1 variants)
| Design | GFLOPS | Notes |
|--------|--------|-------|
| One thread per element (Fig 3.11) | 366.53 | Best parallelism, balanced work |
| One thread per column (3.1b) | 58.09 | Moderate — column writes are strided |
| One thread per row (3.1a) | 18.19 | Worst — poor parallelism per thread |

The one-element-per-thread design dominates because it maximizes thread-level
parallelism and keeps per-thread work balanced.
