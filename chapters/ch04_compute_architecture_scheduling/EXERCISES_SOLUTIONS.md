# Chapter 4: Compute Architecture and Scheduling - Exercise Solutions

**Book:** Programming Massively Parallel Processors (Kirk, Hwu & El Hajj, 4th ed.)
**Hardware:** NVIDIA GeForce RTX 4090, sm_89 (Ada Lovelace), 128 SMs, 24 GB VRAM

## Overview
Chapter 4 covers GPU compute architecture, block scheduling, warps, SIMD execution,
control divergence, warp scheduling, latency tolerance, resource partitioning, and
occupancy. This chapter is conceptual - no CUDA code figures to implement, but the
exercises test understanding of how the GPU hardware actually executes kernels.

## Exercise 1: Kernel Analysis

The kernel from the exercise:
```cuda
__global__ void foo(float* data, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;       // line 02
    if (tid < n) {                                         // line 03
        data[tid] = data[tid] * 2.0f;                      // line 04
        for (int i = 0; i < 100; i++) {                    // line 05
            data[tid] = data[tid] + 1.0f;                  // line 06
        }                                                 // line 07
    }                                                     // line 08
}                                                          // line 09
```

Host function:
```cuda
void callFoo(float* data, int n) {
    int threadsPerBlock = 128;
    int blocksPerGrid = 8;
    foo<<<blocksPerGrid, threadsPerBlock>>>(data, n);
}
```

**Grid configuration:** 8 blocks x 128 threads = 1024 total threads
**Warp size:** 32 threads (standard for all modern GPUs)

### a. Number of warps per block
128 threads / 32 threads per warp = **4 warps per block**

### b. Number of warps in the grid
4 warps/block x 8 blocks = **32 warps in the grid**

### c. For the statement on line 04 (`data[tid] = data[tid] * 2.0f`):

This is inside the `if (tid < n)` guard. Assuming n = 1024 (matching grid size):

**i. How many warps in the grid are active?**
All 1024 threads have tid < 1024, so all threads execute line 04.
**Answer: 32 warps active** (all warps in the grid)

**ii. How many warps in the grid are divergent?**
Since all threads execute the same path (tid < n is true for all), no warp has
threads taking different paths.
**Answer: 0 warps divergent**

**iii. SIMD efficiency of warp 0 of block 0?**
Warp 0 of block 0 covers threads 0-31 (tid = 0-31). All 32 threads are active.
**Answer: 100%**

**iv. SIMD efficiency of warp 1 of block 0?**
Warp 1 of block 0 covers threads 32-63 (tid = 32-63). All 32 threads are active.
**Answer: 100%**

**v. SIMD efficiency of warp 3 of block 0?**
Warp 3 of block 0 covers threads 96-127 (tid = 96-127). All 32 threads are active.
**Answer: 100%**

### d. For the statement on line 07 (end of for loop):

**i. How many warps in the grid are active?**
Same as line 04 - all threads inside the if block reach line 07.
**Answer: 32 warps active**

**ii. How many warps in the grid are divergent?**
The for loop runs exactly 100 iterations for every thread - no early exit, no
conditional break. All threads in every warp execute the same number of iterations.
**Answer: 0 warps divergent**

**iii. SIMD efficiency of warp 0 of block 0?**
All 32 threads execute the loop identically.
**Answer: 100%**

### e. For the loop on line 05:

**i. How many iterations have no divergence?**
All 100 iterations execute identically across all threads - no divergence at all.
**Answer: 100 iterations with no divergence**

**ii. How many iterations have divergence?**
**Answer: 0 iterations with divergence**

---

## Exercise 2: Vector Addition Grid Size

**Given:** Vector length = 2000, 1 thread per element, block size = 512 threads

**Blocks needed:** ceil(2000 / 512) = ceil(3.906) = 4 blocks
**Total threads:** 4 blocks x 512 threads = **2048 threads in the grid**

Note: 48 threads (2048 - 2000) will be inactive due to the boundary check.

---

## Exercise 3: Divergence from Boundary Check

**Given:** 2048 threads in grid, 2000 elements, block size = 512

Threads 2000-2047 (48 threads) will fail the `if (tid < n)` check.
These 48 threads are all in the last block (block 3, threads 448-511).

Warp breakdown of block 3:
- Warp 0: threads 448-479 (tid 448-479) - all active
- Warp 1: threads 480-511 (tid 480-511) - but tid 480-499 active, tid 500-511 inactive

Wait, let me recalculate. Block 3 has threads 1536-2047 in global indexing.
- Threads 1536-1999 (464 threads) are active
- Threads 2000-2047 (48 threads) are inactive

In block 3, threadIdx ranges from 0-511:
- threadIdx 0-463 (tid 1536-1999): active
- threadIdx 464-511 (tid 2000-2047): inactive

Warp breakdown (32 threads each):
- Warp 0: threadIdx 0-31 (tid 1536-1567) - all active
- Warp 1: threadIdx 32-63 (tid 1568-1599) - all active
- ... (warps 2-13: threadIdx 64-447, tid 1600-1983) - all active
- Warp 14: threadIdx 448-479 (tid 1984-2015) - threadIdx 448-463 active (16 threads), 464-479 inactive (16 threads) -> DIVERGENT
- Warp 15: threadIdx 480-511 (tid 2016-2047) - all inactive

**Answer: 1 warp has divergence** (warp 14 of block 3)

---

## Exercise 4: Barrier Wait Time

**Given:** 8 threads with execution times: 2.0, 2.3, 3.0, 2.8, 2.4, 1.9, 2.6, 2.9 us

All threads wait at the barrier until the slowest thread (3.0 us) completes.

| Thread | Execution | Wait at barrier | Total time |
|--------|-----------|-----------------|------------|
| T1     | 2.0       | 1.0             | 3.0        |
| T2     | 2.3       | 0.7             | 3.0        |
| T3     | 3.0       | 0.0             | 3.0        |
| T4     | 2.8       | 0.2             | 3.0        |
| T5     | 2.4       | 0.6             | 3.0        |
| T6     | 1.9       | 1.1             | 3.0        |
| T7     | 2.6       | 0.4             | 3.0        |
| T8     | 2.9       | 0.1             | 3.0        |

Total execution time across all threads: 2.0 + 2.3 + 3.0 + 2.8 + 2.4 + 1.9 + 2.6 + 2.9 = 19.9 us
Total wait time across all threads: 1.0 + 0.7 + 0.0 + 0.2 + 0.6 + 1.1 + 0.4 + 0.1 = 4.1 us
Total time (execution + wait): 19.9 + 4.1 = 24.0 us

**Percentage waiting:** (4.1 / 24.0) x 100 = **17.1%**

---

## Exercise 5: __syncthreads() with 32 threads per block

**Claim:** "If I launch with only 32 threads per block, I can leave out __syncthreads()."

**Answer: NO, this is a BAD idea.**

Reasons:
1. **Hardware guarantee:** __syncthreads() is a barrier synchronization primitive that
   ensures all threads in a block have reached that point before any proceed. While it's
   true that a single warp (32 threads) executes in lockstep on SIMD hardware, the CUDA
   programming model does NOT guarantee this. The hardware may issue instructions out of
   order or use independent thread scheduling (Volta+ architecture).

2. **Independent thread scheduling:** Starting with Volta architecture, threads within
   a warp can be scheduled independently. Even within a single warp, threads can diverge
   and take different execution paths without the warp executing multiple passes.

3. **Portability:** Code that omits __syncthreads() will break on different GPU architectures
   or with different compiler optimizations. The CUDA programming model requires explicit
   synchronization.

4. **Compiler optimization:** The compiler may reorder instructions around __syncthreads()
   barriers. Without the barrier, the compiler has no guarantee of execution order.

---

## Exercise 6: Maximizing SM Threads

**Given:** SM limits: 1536 threads max, 4 blocks max per SM

| Config | Threads/block | Blocks that fit | Total threads |
|--------|--------------|-----------------|---------------|
| a      | 128          | min(4, 1536/128=12) = 4 | 512   |
| b      | 256          | min(4, 1536/256=6) = 4  | 1024  |
| c      | 512          | min(4, 1536/512=3) = 3  | 1536  |
| d      | 1024         | min(4, 1536/1024=1.5) = 1 | 1024 |

**Answer: c. 512 threads per block** gives 1536 threads (maximum possible)

---

## Exercise 7: Occupancy Calculations

**Given:** SM limits: 64 blocks max, 2048 threads max per SM

| Config | Blocks | Threads/block | Total threads | Blocks OK? | Threads OK? | Possible? | Occupancy |
|--------|--------|--------------|---------------|------------|-------------|-----------|-----------|
| a      | 8      | 128          | 1024          | 8 <= 64 Y  | 1024 <= 2048 Y | YES | 1024/2048 = 50% |
| b      | 16     | 64           | 1024          | 16 <= 64 Y | 1024 <= 2048 Y | YES | 1024/2048 = 50% |
| c      | 32     | 32           | 1024          | 32 <= 64 Y | 1024 <= 2048 Y | YES | 1024/2048 = 50% |
| d      | 64     | 32           | 2048          | 64 <= 64 Y | 2048 <= 2048 Y | YES | 2048/2048 = 100% |
| e      | 32     | 64           | 2048          | 32 <= 64 Y | 2048 <= 2048 Y | YES | 2048/2048 = 100% |

**All configurations are possible.** Occupancy levels:
- a: 50%, b: 50%, c: 50%, d: 100%, e: 100%

---

## Exercise 8: Full Occupancy Analysis

**Given:** SM limits: 2048 threads, 32 blocks, 65,536 registers per SM

### a. 128 threads/block, 30 registers/thread

- Blocks for full occupancy: 2048/128 = 16 blocks (16 <= 32, OK)
- Registers per block: 128 x 30 = 3,840
- Total registers: 16 x 3,840 = 61,440 (61,440 <= 65,536, OK)

**Answer: YES, full occupancy achievable. No limiting factor.**

### b. 32 threads/block, 29 registers/thread

- Blocks for full occupancy: 2048/32 = 64 blocks (64 > 32, BLOCKED)
- Max blocks: 32, max threads: 32 x 32 = 1,024
- Registers per block: 32 x 29 = 928
- Total registers: 32 x 928 = 29,696 (well under limit)

**Answer: NO. Limiting factor: maximum blocks per SM (32). Occupancy = 1024/2048 = 50%**

### c. 256 threads/block, 34 registers/thread

- Blocks for full occupancy: 2048/256 = 8 blocks (8 <= 32, OK)
- Registers per block: 256 x 34 = 8,704
- Total registers: 8 x 8,704 = 69,632 (69,632 > 65,536, BLOCKED)
- Max blocks by registers: floor(65,536 / 8,704) = 7 blocks
- Max threads: 7 x 256 = 1,792

**Answer: NO. Limiting factor: registers per SM. Occupancy = 1792/2048 = 87.5%**

---

## Exercise 9: Matrix Multiplication Feasibility

**Given:** 1024x1024 matrix multiply, 32x32 thread blocks, 512 threads/block max,
8 blocks/SM max, 1 thread per output element.

**Analysis:**
- Each 32x32 block = 1,024 threads per block
- 1,024 > 512 threads/block limit -> **EXCEEDS HARDWARE LIMIT**

Even if we ignore the thread limit:
- Output matrix: 1024 x 1024 = 1,048,576 elements
- Threads needed: 1,048,576
- Blocks needed: 1,048,576 / 1,024 = 1,024 blocks
- Blocks per SM: 8
- Total SMs on RTX 4090: 128
- Max concurrent blocks: 128 x 8 = 1,024

So all 1,024 blocks could fit on 128 SMs, but the fundamental problem is:
**32x32 = 1,024 threads per block exceeds the 512 thread limit.**

**Reaction:** The kernel would fail to launch because each block has 1,024 threads,
which exceeds the device's maximum of 512 threads per block. The student needs to
use smaller blocks (e.g., 16x16 = 256 threads) or restructure the kernel.

---

## Key Concepts Summary

| Concept | Description |
|---------|-------------|
| Warp | Group of 32 threads executed in SIMD fashion |
| Block scheduling | Blocks assigned to SMs arbitrarily; threads in different blocks cannot synchronize |
| Control divergence | Threads in same warp take different execution paths; reduces SIMD efficiency |
| Occupancy | Ratio of active threads to max threads per SM; higher = better latency hiding |
| Resource limits | Threads, blocks, registers, shared memory per SM constrain occupancy |
| Transparent scalability | Kernel works regardless of grid size; blocks scheduled as SMs become available |
