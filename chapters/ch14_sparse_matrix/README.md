# Chapter 14: Sparse Matrix Computation

Implementations of sparse matrix-vector multiplication (SpMV) kernels using five different storage formats, plus a COO-to-CSR conversion utility (Exercise 3).

## Kernels

| Binary | Source | Description | Format |
|--------|--------|-------------|--------|
| `ch14_spmv_coo` | `ch14_spmv_coo.cu` | Fig 14.5 — One thread per nonzero, atomicAdd | COO |
| `ch14_spmv_csr` | `ch14_spmv_csr.cu` | Fig 14.9 — One thread per row, no atomics | CSR |
| `ch14_spmv_ell` | `ch14_spmv_ell.cu` | Fig 14.12 — Column-major, coalesced | ELL |
| `ch14_spmv_hybrid_ell_coo` | `ch14_spmv_hybrid_ell_coo.cu` | Fig 14.13 — ELL capped at K/row + COO overflow | Hybrid ELL-COO |
| `ch14_spmv_jds` | `ch14_spmv_jds.cu` | Exercise 5 — Sorted rows, minimal divergence | JDS |
| `ch14_coo_to_csr` | `ch14_coo_to_csr.cu` | Exercise 3 — Histogram + prefix sum + scatter | Conversion |

## Storage Format Comparison

| Format | Space Efficiency | Memory Access | Atomics | Control Divergence | Flexibility |
|--------|:----------------:|:-------------:|:-------:|:------------------:|:-----------:|
| COO | Low (3 arrays) | Coalesced | Yes (atomicAdd) | None (uniform work) | High (append-friendly) |
| CSR | High (rowPtrs replace rowIdx) | NOT coalesced | No | High (variable row lengths) | Low (insertion costly) |
| ELL | Low (padding overhead) | Coalesced | No | High (same as CSR) | Medium (replace padding) |
| Hybrid ELL-COO | Medium (reduced padding) | Coalesced | For COO part only | Reduced | High (ELL + COO append) |
| JDS | High (no padding) | Coalesced | No | Low (similar-length rows in same warp) | Very low (sorting needed) |

## Performance Results

### Test: 4×4 Book Example (Fig 14.1 matrix)

| Kernel | Validation | Result |
|--------|:----------:|:------:|
| SpMV/COO | PASS | y = [22, 24, 17, 6] |
| SpMV/CSR | PASS | y = [22, 24, 17, 6] |
| SpMV/ELL | PASS | y = [22, 24, 17, 6] |
| SpMV/JDS | PASS | y = [22, 24, 17, 6] |
| COO→CSR | PASS | rowPtrs = [0, 2, 3, 5, 7] |

### Throughput on GTX 1050

| Kernel | Matrix | Nonzeros | Time | Throughput |
|--------|--------|:--------:|:----:|:----------:|
| SpMV/COO | 1024×1024, 1% density | 10,718 | 0.012 ms | 905 M nz/s |
| SpMV/CSR | 4096×4096, 0.5% density | 83,909 | 0.040 ms | 2,084 M nz/s |
| SpMV/ELL | 4096×4096, 0.5% density | 83,659 | 0.034 ms | 2,483 M nz/s |
| SpMV/JDS | 1024×1024, variable dist. | 4,994 | 0.012 ms | 414 M nz/s |

*Note: Throughput depends heavily on the matrix size, density distribution, and row-length variance. The GTX 1050 (sm_61, 5 SMs) has limited memory bandwidth; results scale with problem size.*

### Hybrid ELL-COO Padding Reduction

For a 512×512 skewed matrix (one row with 20 nonzeros, others with ~3):

| Format | Storage Slots | Overhead |
|--------|:-------------:|:--------:|
| Full ELL (K=20) | 10,240 | 167% padding |
| Hybrid ELL-COO (K=4, ELL + COO) | 2,048 + 1,830 | ~1% (only COO overhead) |

## Key Observations

1. **COO is simplest** but uses atomicAdd → contention on dense rows
2. **CSR is space-efficient** but accesses are uncoalesced → lower effective bandwidth
3. **ELL is coalesced** but padding overhead can be extreme for skewed distributions
4. **Hybrid ELL-COO** addresses the padding problem by capping ELL at K nonzeros/row
5. **JDS** achieves coalesced access AND low divergence by sorting rows by length — at the cost of sorting overhead and reduced flexibility
6. **COO→CSR conversion** (Exercise 3) combines three fundamental primitives: histogram + prefix sum + scatter

## Files

```
ch14_sparse_matrix/
├── ch14_spmv_coo.cu           # SpMV/COO kernel (Fig 14.5)
├── ch14_spmv_csr.cu           # SpMV/CSR kernel (Fig 14.9)
├── ch14_spmv_ell.cu           # SpMV/ELL kernel (Fig 14.12)
├── ch14_spmv_hybrid_ell_coo.cu # Hybrid ELL-COO (Fig 14.13)
├── ch14_spmv_jds.cu           # SpMV/JDS kernel (Exercise 5)
├── ch14_coo_to_csr.cu         # COO→CSR conversion (Exercise 3)
├── README.md                  # This file
└── (binaries)                 # Compiled outputs
```

## Building & Running

```bash
nvcc -std=c++17 -arch=sm_61 -O2 -o ch14_spmv_coo ch14_spmv_coo.cu && ./ch14_spmv_coo
```
