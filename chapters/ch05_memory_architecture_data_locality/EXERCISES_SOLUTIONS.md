# Chapter 5: Memory Architecture and Data Locality - Exercise Solutions

## Exercise 1: Matrix addition and shared memory

**Question:** Can one use shared memory to reduce the global memory bandwidth consumption for matrix addition?

**Answer:** No. In matrix addition C[i,j] = A[i,j] + B[i,j], each thread reads exactly one element from A and one from B, and writes one element to C. There is no overlap in the elements accessed by different threads -- each element of A and B is read exactly once. Shared memory is useful when multiple threads within a block need to access the same data. Since matrix addition has no data reuse between threads, shared memory provides no benefit.

## Exercise 2: Tiling diagrams for 8x8 matrix multiplication

**Question:** Draw equivalent of Fig 5.7 for 8x8 matmul with 2x2 and 4x4 tiling. Verify reduction is proportional to tile size.

**Answer:**
- With 2x2 tiling: 16 blocks, each computes a 2x2 tile. Global memory traffic reduced by factor of 2.
- With 4x4 tiling: 4 blocks, each computes a 4x4 tile. Global memory traffic reduced by factor of 4.
- Verification: For 2x2 tiles, each thread loads 2 elements of M and 2 of N per phase, 4 phases total = 8 loads. Without tiling, each thread would load 8 elements of M and 8 of N = 16 loads. Reduction = 2x. For 4x4 tiles, 4 phases, 4 loads per phase = 16 loads vs 32 without tiling. Reduction = 4x. Confirmed.

## Exercise 3: Missing __syncthreads() consequences

**Question:** What incorrect behavior can happen if one forgot __syncthreads()?

**Answer:**
- Missing first __syncthreads() (after loading): Some threads may start computing the dot product before all tiles are loaded. They would use stale data from previous phase or uninitialized shared memory, producing incorrect results.
- Missing second __syncthreads() (after computing): Some threads may start loading the next tile before others finish reading the current tile. The shared memory would be overwritten with new data while other threads still need the old data for their dot product calculation.
- Both cases produce race conditions that corrupt the computation.

## Exercise 4: Shared memory vs registers for holding fetched values

**Question:** Why use shared memory instead of registers to hold values fetched from global memory?

**Answer:** Shared memory is shared within a block, so data loaded once can be reused by all threads in the block. If each thread loaded the same data into its private registers, the global memory bandwidth would be wasted -- the same data would be fetched multiple times (once per thread). Shared memory enables data sharing within a block, reducing redundant global memory traffic. Registers are private to each thread and cannot share data.

## Exercise 5: 32x32 tile memory bandwidth reduction

**Question:** What is the reduction of memory bandwidth usage with a 32x32 tile?

**Answer:** The reduction factor equals the tile dimension. With TILE_WIDTH=32, the reduction is 32x. Each element of M and N is loaded once into shared memory per tile phase and reused by all 32 threads in the corresponding row/column. Without tiling, each of the 32 threads would load the same element independently.

## Exercise 6: Local variable versions

**Question:** 1000 blocks x 512 threads. How many versions of a local variable?

**Answer:** 1000 * 512 = 512,000 versions. Local (automatic) variables have thread scope, so each thread gets its own private copy.

## Exercise 7: Shared variable versions

**Question:** Same setup. How many versions of a shared variable?

**Answer:** 1000 versions. Shared variables have block scope, so each block gets one copy shared by all threads in that block.

## Exercise 8: Global memory accesses for NxN matmul

**Question:** How many times is each element requested from global memory?

**Answer:**
a. Without tiling: Each element of M is read N times (once for each column of N). Each element of N is read N times (once for each row of M). Total: N times per element.
b. With T x T tiles: Each element is loaded once per tile phase. There are N/T tile phases, so each element is loaded N/T times. Total: N/T times per element. Reduction factor = T.

## Exercise 9: Compute-bound or memory-bound?

**Question:** Kernel does 36 FLOP and 7x32-bit global memory accesses per thread.

**Analysis:**
- 36 FLOP / (7 * 4 bytes) = 36/28 = 1.29 OP/B arithmetic intensity

a. Peak FLOPS=200 GFLOPS, peak BW=100 GB/s
   - Memory-bound ceiling: 100 * 1.29 = 129 GFLOPS < 200 GFLOPS
   - **Memory-bound** (limited by bandwidth)

b. Peak FLOPS=300 GFLOPS, peak BW=250 GB/s
   - Memory-bound ceiling: 250 * 1.29 = 322.5 GFLOPS > 300 GFLOPS
   - **Compute-bound** (limited by compute)

## Exercise 10: Tile transpose kernel

**Question:** Kernel transposes tiles of size BLOCK_WIDTH x BLOCK_WIDTH. For what BLOCK_WIDTH values does it work?

**Answer:**
a. The kernel uses `__shared__ float tile[BLOCK_WIDTH][BLOCK_WIDTH]` for shared memory. For sm_89, shared memory per block is 48 KB = 49,152 bytes. The tile needs 2 * BLOCK_WIDTH * BLOCK_WIDTH * 4 bytes (two tiles for read and write). So: 8 * BLOCK_WIDTH^2 <= 49152, giving BLOCK_WIDTH <= 78. But BLOCK_WIDTH is limited to 1-20, so all values work for shared memory capacity.

   However, the real issue is that the kernel likely uses `__syncthreads()` incorrectly or has race conditions. The kernel probably needs proper synchronization between the load and store phases.

b. Root cause: The kernel likely writes to shared memory and then reads from it without proper synchronization, or uses the same shared memory array for both source and destination tiles. Fix: Use separate shared memory arrays for input and output tiles, or add proper `__syncthreads()` barriers between phases.

## Exercise 11: Memory scoping analysis

**Question:** Analyze variable scoping in a kernel.

**Answer:**
a. Variable `i`: Local variable, one per thread. If 256 threads/block and 4 blocks: 1024 versions.
b. Array `x[]`: If declared as local array, it spills to local memory. One per thread: 1024 versions.
c. Variable `y_s`: Shared variable, one per block: 4 versions.
d. Array `b_s[]`: Shared array, one per block: 4 versions.
e. Shared memory per block: Sum of all shared variable sizes. If `b_s[]` is 256 floats: 256 * 4 = 1024 bytes + any other shared vars.
f. FLOP to global memory ratio: Count FLOP operations and global memory accesses. If 256 FLOP and 256 global loads: 256 / (256 * 4) = 0.25 OP/B.

## Exercise 12: Occupancy analysis

**Question:** GPU limits: 2048 threads/SM, 32 blocks/SM, 64K registers/SM, 96 KB shared mem/SM.

**Answer:**
a. 64 threads/block, 27 registers/thread, 4 KB shared mem/SM
   - Threads: 64 threads/block, max 32 blocks/SM = 2048 threads/SM. OK.
   - Registers: 64 * 27 = 1728 registers/block. 32 blocks * 1728 = 55,296 < 65,536. OK.
   - Shared mem: 4 KB/block. 32 blocks * 4 KB = 128 KB > 96 KB. **Shared memory limited.**
   - Max blocks by shared mem: 96/4 = 24 blocks. Max threads: 24 * 64 = 1536.
   - **Occupancy: 1536/2048 = 75%. Limited by shared memory.**

b. 256 threads/block, 31 registers/thread, 8 KB shared mem/SM
   - Threads: 256 threads/block, max 32 blocks/SM = 8192 > 2048. Max 8 blocks for 2048 threads.
   - Registers: 256 * 31 = 7936 registers/block. 8 blocks * 7936 = 63,488 < 65,536. OK.
   - Shared mem: 8 KB/block. 8 blocks * 8 KB = 64 KB < 96 KB. OK.
   - **Full occupancy achievable (8 blocks = 2048 threads).**
