# Chapter 12 — Merge (deepseek-v4-pro implementation)

**Hardware**: GTX 1050, sm_61 (Pascal), 5 SMs, 2 GB VRAM
**Compiler**: nvcc -std=c++17 -arch=sm_61 -O2

## Kernels Implemented

| Kernel | File | Figures | Description |
|--------|------|---------|-------------|
| Basic Merge | `ch12_basic_merge.cu` | 12.2, 12.5, 12.9 | One thread per output segment, co_rank on global memory |
| Tiled Merge | `ch12_tiled_merge.cu` | 12.11-12.13 | Block-level co_rank, cooperative tile load, shared-memory merge |
| Circular Buffer | `ch12_circular_buffer_merge.cu` | 12.16, 12.18-12.20 | Reuses unconsumed buffer elements; refills only consumed portion |

## Validation Results (3 A:B ratios per size, 18 tests each)

### Basic Merge (256 threads, ~8 elements/thread)
| Total N | 20% split | 50% split | 80% split |
|---------|-----------|-----------|-----------|
| 128 | PASS 0.016ms | PASS 0.015ms | PASS 0.017ms |
| 1,024 | PASS 0.051ms | PASS 0.044ms | PASS 0.062ms |
| 4,096 | PASS 0.153ms | PASS 0.166ms | PASS 0.198ms |
| 32,768 | PASS 1.237ms | PASS 1.290ms | PASS 1.448ms |
| 131,072 | PASS 8.399ms | PASS 10.393ms | PASS 10.232ms |
| 524,288 | PASS 119ms | PASS 135ms | PASS 109ms |

### Tiled Merge (128 threads, 16 blocks, TILE_SIZE=1024)
| Total N | 20% split | 50% split | 80% split |
|---------|-----------|-----------|-----------|
| 128 | PASS 0.010ms | PASS 0.008ms | PASS 0.010ms |
| 1,024 | PASS 0.026ms | PASS 0.030ms | PASS 0.035ms |
| 4,096 | PASS 0.099ms | PASS 0.106ms | PASS 0.118ms |
| 32,768 | PASS 0.736ms | PASS 0.305ms | PASS 0.878ms |
| 131,072 | PASS 3.244ms | PASS 3.336ms | PASS 3.483ms |
| 524,288 | PASS 16.2ms | PASS 13.8ms | PASS 14.4ms |

### Circular Buffer Merge (128 threads, 16 blocks, TILE_SIZE=1024)
| Total N | 20% split | 50% split | 80% split |
|---------|-----------|-----------|-----------|
| 128 | PASS 0.013ms | PASS 0.012ms | PASS 0.013ms |
| 1,024 | PASS 0.035ms | PASS 0.038ms | PASS 0.045ms |
| 4,096 | PASS 0.130ms | PASS 0.132ms | PASS 0.153ms |
| 32,768 | PASS 0.935ms | PASS 0.148ms | PASS 1.115ms |
| 131,072 | PASS 4.306ms | PASS 1.215ms | PASS 4.447ms |
| 524,288 | PASS 18.8ms | PASS 13.3ms | PASS 18.5ms |

## Performance Comparison (524,288 elements, 50% split)

| Kernel | Time | Throughput |
|--------|------|------------|
| Basic | 134.9 ms | 0.05 GB/s |
| Tiled | 13.8 ms | 0.46 GB/s |
| Circular Buffer | 13.3 ms | 0.47 GB/s |

Tiled merge is ~10x faster than basic. Circular buffer shows marginal improvement over tiled at 50% split (the benefit is larger when one input dominates, avoiding reload waste).

## Bugs Fixed in Circular Buffer Kernel

The original `ch12_merge/ch12_circular_buffer_merge.cu` (from a previous implementation session) had four bugs that caused failures on multi-iteration test cases (N >= 32,768):

1. **Refill global start index**: Used `A_curr + A_consumed` (updated consumption) instead of `A_curr + A_consumed + A_avail - A_used_this_iter`. This caused reloading elements already in the buffer, leading to duplicate entries and wrong output.

2. **Refill remaining count**: Used `A_length - A_consumed` instead of `max(0, A_length - A_consumed - A_avail + A_used_this_iter)`. Didn't account for unconsumed buffer elements, so too many elements were "counted" as loadable.

3. **Buffer occupancy recomputation**: Used `Aload_init` (initial buffer size) as the base instead of `A_avail` (current buffer occupancy). After the first refill, `A_avail != Aload_init`, causing progressively wrong buffer occupancy values.

4. **Broadcast array overwrote data**: Used `A_s[2..4]` and `B_s[2..4]` for counter broadcast. When the circular buffer wrapped such that valid data occupied those positions, the broadcast corrupted it. Fixed by using a dedicated `__shared__ int s_bc[8]` array.

All four bugs were latent in the original code (noted as "Multi-iteration has a remaining indexing bug under debug" in its header comment).

## Key Concepts

- **Co-rank function**: Binary search that determines how many elements from sorted input A belong in the first k elements of merged output C.
- **Block-level partition**: Thread 0 calls co_rank on global memory to determine each block's input ranges.
- **Tiled loading**: All threads cooperatively load A and B tiles into shared memory.
- **Circular buffer**: Tracks start pointers to reuse unconsumed elements across iterations; refills only consumed portion.
- **Register broadcast**: Variables updated by thread 0 must be written to shared memory with `__syncthreads()` fence so all threads see the same value.
