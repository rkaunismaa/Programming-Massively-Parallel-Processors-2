/*
 * =============================================================================
 *  Chapter 6: Performance Considerations — Corner Turning
 * =============================================================================
 *  Book:      Programming Massively Parallel Processors (Kirk, Hwu & El Hajj,
 *             4th Edition, Morgan Kaufmann 2023)
 *  Section:   6.1 — Corner turning for coalesced access to column-major matrices
 *  Purpose:   Demonstrate the corner turning optimization that transforms
 *             uncoalesced global memory accesses into coalesced ones when one
 *             operand of a matrix multiply is stored in column-major layout.
 *  Hardware:  GTX 1050, sm_61 (Pascal) — all code targets this GPU
 * =============================================================================
 *
 *  THE PROBLEM
 *  ---------------------------------------------------------------------------
 *  In standard tiled matrix multiplication (Ch. 5), both input matrices are
 *  assumed to be row-major.  When matrix B is stored in column-major layout
 *  (e.g., Fortran, MATLAB, NumPy with order='F'), the shared-memory load
 *  pattern for B becomes catastrophically uncoalesced.
 *
 *  Consider thread (tx, ty) in block (bx, by) loading element B[row_b][col_b]:
 *    row_b = ph * TILE_WIDTH + ty
 *    col_b = bx * TILE_WIDTH + tx
 *
 *  In column-major storage, B[row_b][col_b] lives at:
 *    address = col_b * width + row_b
 *           = (bx*TILE_WIDTH + tx) * width + (ph*TILE_WIDTH + ty)
 *
 *  Two threads that differ only in tx (i.e., consecutive threads in the same
 *  warp) access addresses that differ by `width` bytes — not by sizeof(float).
 *  For width=2048, consecutive warp threads are 2048 floats apart in memory.
 *  This means every warp issues a single memory transaction that spans ~8 MB
 *  of address space, hitting cache lines all over the place.
 *
 *
 *  CORNER TURNING — THE SOLUTION (Figure 6.1 / Section 6.1)
 *  ---------------------------------------------------------------------------
 *  The key insight: instead of having thread (tx, ty) load B[row_b][col_b]
 *  into shared memory position [ty][tx], we swap the roles of tx and ty
 *  for the B tile load:
 *
 *    thread (tx, ty) loads B[row_b][col_b] into shared memory position [tx][ty]
 *    where  row_b = ph*TILE_WIDTH + tx    (note: tx, not ty)
 *           col_b = bx*TILE_WIDTH + ty    (note: ty, not tx)
 *
 *  In column-major storage:
 *    address = col_b * width + row_b
 *           = (bx*TILE_WIDTH + ty) * width + (ph*TILE_WIDTH + tx)
 *
 *  Now consecutive threads (differing in tx by 1) access addresses differing
 *  by exactly 1 float — perfectly coalesced!
 *
 *  The shared memory tile is effectively "turned" (transposed) relative to
 *  the standard approach.  This is why it's called "corner turning" — the
 *  data is loaded along the other diagonal of the tile.
 *
 *  When computing the dot product, we account for the transposition:
 *    Standard:  Pvalue += A_s[ty][k] * B_s[k][tx]
 *    Corner:   Pvalue += A_s[ty][k] * B_s[k][tx]
 *
 *  Wait — the dot product access pattern is identical because:
 *    - We need B[row_k][col] for k=0..TILE_WIDTH-1
 *    - row_k = ph*TILE_WIDTH + k
 *    - col   = bx*TILE_WIDTH + tx
 *    - In corner turning, B_s[tx][ty] = B[ph*TILE_WIDTH+tx][bx*TILE_WIDTH+ty]
 *    - So B_s[k][tx] = B[ph*TILE_WIDTH+k][bx*TILE_WIDTH+tx]  -- correct!
 *
 *  The "corner turn" is purely in the loading phase.  The shared memory
 *  layout is transposed, but the dot product accesses B_s[k][tx] which
 *  still gives the right element because both indices swap together.
 *
 *
 *  WHY THIS WORKS FOR ANY B LAYOUT
 *  ---------------------------------------------------------------------------
 *  Corner turning makes the B tile load coalesced regardless of whether B
 *  is row-major or column-major — but for opposite reasons:
 *
 *  Row-major B:  address = row_b * width + col_b
 *               = (ph*TILE_WIDTH + tx) * width + (bx*TILE_WIDTH + ty)
 *  Consecutive tx differ by `width` — NOT coalesced in row-major.
 *  But the standard load IS coalesced for row-major, so corner turning
 *  is only beneficial when B is column-major.
 *
 *  Column-major B: address = col_b * width + row_b
 *                = (bx*TILE_WIDTH + ty) * width + (ph*TILE_WIDTH + tx)
 *  Consecutive tx differ by 1 — COALESCED for column-major.
 *
 *  In practice, corner turning is applied when you know B is column-major
 *  (or more generally, when the memory layout of B makes the standard
 *  load pattern uncoalesced).
 *
 *
 *  SHARED MEMORY LAYOUT DIAGRAM
 *  ---------------------------------------------------------------------------
 *  Standard tiled matmul (row-major B):
 *
 *  Thread (tx,ty) loads:          Shared memory B_s:
 *    B[ty + ph*TILE][tx + bx*TILE]
 *
 *  B_s[0][0]  B_s[0][1]  B_s[0][2]  ...
 *  B_s[1][0]  B_s[1][1]  B_s[1][2]  ...
 *  B_s[2][0]  B_s[2][1]  B_s[2][2]  ...
 *  ...
 *
 *  Corner turning (column-major B):
 *
 *  Thread (tx,ty) loads:          Shared memory B_s:
 *    B[tx + ph*TILE][ty + bx*TILE]   (note tx/ty swap in indices)
 *
 *  B_s[0][0]  B_s[0][1]  B_s[0][2]  ...   (row 0 of B_s = column 0 of B tile)
 *  B_s[1][0]  B_s[1][1]  B_s[1][2]  ...   (row 1 of B_s = column 1 of B tile)
 *  B_s[2][0]  B_s[2][1]  B_s[2][2]  ...   (row 2 of B_s = column 2 of B tile)
 *  ...
 *
 *  B_s is the TRANSPOSE of the B tile.  The dot product still works because
 *  B_s[k][tx] = B[row_k][col] with row_k = ph*TILE+tx, col = bx*TILE+tx
 *  ... wait, let me be precise:
 *
 *  B_s[tx][ty] = B[ph*TILE + tx][bx*TILE + ty]
 *  B_s[k][tx]  = B[ph*TILE + k][bx*TILE + tx]   <-- this is B[row_k][col]
 *  where row_k = ph*TILE + k and col = bx*TILE + tx.  Correct!
 * =============================================================================
 */

#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstdlib>
#include "../common/cuda_utils.cuh"

#define TILE_WIDTH    16

/*
 * Corner-turning tiled matrix multiplication kernel.
 *
 * Computes P = A * B where:
 *   - A is stored in ROW-MAJOR order (standard C layout)
 *   - B is stored in COLUMN-MAJOR order (Fortran/MATLAB layout)
 *   - P is stored in ROW-MAJOR order
 *
 * The corner turning technique ensures coalesced global memory accesses
 * for both A and B tiles.
 *
 * Parameters:
 *   d_A     - matrix A (width x width), row-major: A[i*width + j]
 *   d_B_col - matrix B (width x width), column-major: B[j*width + i]
 *   d_P     - output P = A * B (width x width), row-major
 *   width   - dimension of the square matrices
 *
 * Requirements:
 *   width must be divisible by TILE_WIDTH
 */
__global__ void matmulCornerTurning(const float* __restrict__ d_A,
                                    const float* __restrict__ d_B_col,
                                    float* __restrict__ d_P,
                                    int width)
{
    // -----------------------------------------------------------------------
    // Shared memory tiles
    // -----------------------------------------------------------------------
    // A_s is loaded normally: A_s[ty][tx] = A[row][col]
    // B_s is loaded with corner turning: B_s[tx][ty] = B[row][col]
    //   This means B_s is the TRANSPOSE of the B tile in shared memory.
    __shared__ float A_s[TILE_WIDTH][TILE_WIDTH];
    __shared__ float B_s[TILE_WIDTH][TILE_WIDTH];

    // -----------------------------------------------------------------------
    // Thread and block indices
    // -----------------------------------------------------------------------
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // -----------------------------------------------------------------------
    // Output element coordinates
    // -----------------------------------------------------------------------
    // Each block (bx, by) is responsible for TILE_WIDTH x TILE_WIDTH elements
    // of the output matrix P.
    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    // -----------------------------------------------------------------------
    // Accumulator for the dot product
    // -----------------------------------------------------------------------
    float Pvalue = 0.0f;

    // -----------------------------------------------------------------------
    // Loop over tiles along the reduction (k) dimension
    // -----------------------------------------------------------------------
    // ph indexes which tile pair (A_tile, B_tile) we are processing
    for (int ph = 0; ph < width / TILE_WIDTH; ++ph) {

        // ================================================================
        // 1. Load A tile — STANDARD (coalesced for row-major A)
        // ================================================================
        // Thread (tx, ty) loads A[row][col] where:
        //   row = by*TILE_WIDTH + ty        (fixed for this thread)
        //   col = ph*TILE_WIDTH + tx        (advances by TILE_WIDTH each ph)
        // In row-major: address = row*width + col
        // Consecutive tx -> consecutive addresses -> COALESCED
        A_s[ty][tx] = d_A[row * width + ph * TILE_WIDTH + tx];

        // ================================================================
        // 2. Load B tile — CORNER TURNING (coalesced for column-major B)
        // ================================================================
        //
        // STANDARD approach (would be UNCOALESCED for column-major B):
        //   B_s[ty][tx] = B_col[(ph*TILE+ty)*width + (bx*TILE+tx)]
        //   Thread tx accesses column (bx*TILE+tx) of row (ph*TILE+ty)
        //   In column-major: address = (bx*TILE+tx)*width + (ph*TILE+ty)
        //   Consecutive tx -> address differs by width -> NOT coalesced
        //
        // CORNER TURNING approach (COALESCED for column-major B):
        //   B_s[tx][ty] = B_col[(bx*TILE+ty)*width + (ph*TILE+tx)]
        //   Thread tx accesses column (bx*TILE+ty) of row (ph*TILE+tx)
        //   Note: ty is FIXED for this thread, tx varies
        //   In column-major: address = (bx*TILE+ty)*width + (ph*TILE+tx)
        //   Consecutive tx -> address differs by 1 -> COALESCED
        //
        // The key swap: tx selects the ROW in B (not the column),
        // and ty selects the COLUMN in B (not the row).
        // This is the "corner turn" — we traverse the tile along
        // the other diagonal.
        B_s[tx][ty] = d_B_col[(bx * TILE_WIDTH + ty) * width + (ph * TILE_WIDTH + tx)];

        // -----------------------------------------------------------------------
        // Synchronize: wait for all threads to finish loading both tiles
        // -----------------------------------------------------------------------
        __syncthreads();

        // ================================================================
        // 3. Compute partial dot product
        // ================================================================
        //
        // We need: Pvalue += sum over k of A[row][k] * B[k][col]
        // where k ranges over the current tile: k = ph*TILE + k_local
        //
        // A_s[ty][k] = A[row][ph*TILE + k]       (standard load, correct)
        // B_s[k][tx] = B[ph*TILE + k][bx*TILE + tx]  (corner-turned load)
        //
        // Let's verify B_s[k][tx]:
        //   B_s[tx'][ty'] = B[(bx*TILE+ty')*width + (ph*TILE+tx')]  in column-major
        //   B_s[k][tx]    = B[ph*TILE + k][bx*TILE + tx]
        //   This is B[row_k][col] where row_k = ph*TILE+k, col = bx*TILE+tx. CORRECT!
        //
        // The dot product access pattern is IDENTICAL to the standard version
        // because the transposition in loading is symmetric — both indices
        // swap together, so B_s[k][tx] still gives B[row_k][col].
        for (int k = 0; k < TILE_WIDTH; ++k) {
            Pvalue += A_s[ty][k] * B_s[k][tx];
        }

        // -----------------------------------------------------------------------
        // Synchronize: wait before loading the next tile pair
        // This ensures no thread reads from A_s/B_s while another is writing
        // -----------------------------------------------------------------------
        __syncthreads();
    }

    // -----------------------------------------------------------------------
    // Write the final result to global memory
    // -----------------------------------------------------------------------
    d_P[row * width + col] = Pvalue;
}


/*
 * CPU reference: matrix multiplication where B is in column-major storage.
 *
 * Computes P = A * B by reading B in column-major order.
 * B_col[j*width + i] corresponds to B[i][j] in mathematical notation.
 */
void cpu_matmul_colmajor(const float* A, const float* B_col, float* P, int width)
{
    for (int i = 0; i < width; ++i) {
        for (int j = 0; j < width; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < width; ++k) {
                // A is row-major:   A[i][k] = A[i*width + k]
                // B is column-major: B[k][j] = B_col[j*width + k]
                sum += A[i * width + k] * B_col[j * width + k];
            }
            P[i * width + j] = sum;
        }
    }
}


/*
 * Convert a row-major matrix to column-major.
 * B_col[j*width + i] = B_row[i*width + j]
 */
void row_to_colmajor(const float* B_row, float* B_col, int width)
{
    for (int i = 0; i < width; ++i) {
        for (int j = 0; j < width; ++j) {
            B_col[j * width + i] = B_row[i * width + j];
        }
    }
}


int main()
{
    // -----------------------------------------------------------------------
    // Configuration
    // -----------------------------------------------------------------------
    const int width = 2048;
    const int bytes = width * width * sizeof(float);

    std::cout << "============================================================" << std::endl;
    std::cout << "Chapter 6: Corner Turning — Coalesced Access for Column-Major B" << std::endl;
    std::cout << "============================================================" << std::endl;
    std::cout << "Matrix size     : " << width << " x " << width << std::endl;
    std::cout << "Tile size       : " << TILE_WIDTH << " x " << TILE_WIDTH << std::endl;
    std::cout << "A layout        : Row-major" << std::endl;
    std::cout << "B layout        : Column-major" << std::endl;
    std::cout << "P layout        : Row-major" << std::endl;
    std::cout << std::endl;

    // -----------------------------------------------------------------------
    // Select GPU — GTX 1050 on device 1
    // -----------------------------------------------------------------------
    CHECK_CUDA(cudaSetDevice(1));

    // Print device information
    print_device_info(1);

    // -----------------------------------------------------------------------
    // Allocate host memory
    // -----------------------------------------------------------------------
    float* h_A       = new float[width * width];
    float* h_B_row   = new float[width * width];  // B in row-major (logical)
    float* h_B_col   = new float[width * width];  // B in column-major (device)
    float* h_P_gpu   = new float[width * width];
    float* h_P_cpu   = new float[width * width];

    // -----------------------------------------------------------------------
    // Initialize matrices with deterministic values
    // -----------------------------------------------------------------------
    srand(42);
    for (int i = 0; i < width * width; ++i) {
        h_A[i]     = static_cast<float>(rand() % 100) / 100.0f;
        h_B_row[i] = static_cast<float>(rand() % 100) / 100.0f;
    }

    // Convert B from row-major (logical) to column-major (device storage)
    row_to_colmajor(h_B_row, h_B_col, width);

    // -----------------------------------------------------------------------
    // Allocate device memory
    // -----------------------------------------------------------------------
    const float *d_A, *d_B_col;
    float* d_P;
    CHECK_CUDA(cudaMalloc((void**)&d_A, bytes));
    CHECK_CUDA(cudaMalloc((void**)&d_B_col, bytes));
    CHECK_CUDA(cudaMalloc((void**)&d_P, bytes));

    // Copy input data to device
    CHECK_CUDA(cudaMemcpy((void*)d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy((void*)d_B_col, h_B_col, bytes, cudaMemcpyHostToDevice));

    // -----------------------------------------------------------------------
    // Configure kernel launch parameters
    // -----------------------------------------------------------------------
    // Block: TILE_WIDTH x TILE_WIDTH = 16 x 16 = 256 threads
    // Grid:  (width/TILE_WIDTH) x (width/TILE_WIDTH) = 128 x 128 blocks
    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid(width / TILE_WIDTH, width / TILE_WIDTH);

    std::cout << "Grid  : " << grid.x << " x " << grid.y << " blocks" << std::endl;
    std::cout << "Block : " << block.x << " x " << block.y << " threads" << std::endl;
    std::cout << "Threads per block : " << (block.x * block.y) << std::endl;
    std::cout << "Total blocks      : " << (grid.x * grid.y) << std::endl;
    std::cout << std::endl;

    // -----------------------------------------------------------------------
    // Warmup run — ensures kernel is JIT-compiled and caches are populated
    // -----------------------------------------------------------------------
    matmulCornerTurning<<<grid, block>>>(d_A, d_B_col, d_P, width);
    CHECK_CUDA(cudaDeviceSynchronize());

    // -----------------------------------------------------------------------
    // Timed run
    // -----------------------------------------------------------------------
    gpu_timer timer;
    timer.start();
    matmulCornerTurning<<<grid, block>>>(d_A, d_B_col, d_P, width);
    timer.stop();
    float elapsed_ms = timer.elapsed_ms();

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_P_gpu, d_P, bytes, cudaMemcpyDeviceToHost));

    // -----------------------------------------------------------------------
    // Performance output
    // -----------------------------------------------------------------------
    std::cout << "------------------------------------------------------------" << std::endl;
    std::cout << "Performance Results:" << std::endl;
    std::cout << "  Kernel time   : " << std::fixed << std::setprecision(2)
              << elapsed_ms << " ms" << std::endl;

    // GFLOPS = 2 * N^3 FLOPs / (time in seconds) / 1e9
    // Factor of 2: each multiply-accumulate counts as 2 FLOPs (mul + add)
    double flops = 2.0 * width * width * width;
    double gflops = (flops / (elapsed_ms * 1e6));
    std::cout << "  GFLOPS        : " << std::fixed << std::setprecision(2)
              << gflops << " GFLOPS" << std::endl;
    std::cout << "  Total FLOPs   : " << std::fixed << std::setprecision(0)
              << flops << std::endl;
    std::cout << "------------------------------------------------------------" << std::endl;

    // -----------------------------------------------------------------------
    // Validation: CPU reference with column-major B
    // -----------------------------------------------------------------------
    std::cout << std::endl;
    std::cout << "Validation (CPU reference with column-major B)..." << std::endl;
    cpu_matmul_colmajor(h_A, h_B_col, h_P_cpu, width);

    bool passed = cpu_allclose(h_P_cpu, h_P_gpu, width * width, 1.0f);
    std::cout << "  Result: " << (passed ? "PASSED" : "FAILED") << std::endl;
    std::cout << std::endl;

    // -----------------------------------------------------------------------
    // Corner turning explanation summary
    // -----------------------------------------------------------------------
    std::cout << "============================================================" << std::endl;
    std::cout << "Corner Turning Summary:" << std::endl;
    std::cout << "------------------------------------------------------------" << std::endl;
    std::cout << "  Without corner turning (column-major B):" << std::endl;
    std::cout << "    Thread tx loads B at address differing by `width`" << std::endl;
    std::cout << "    from its neighbor -> UNCOALESCED (stride-2048)" << std::endl;
    std::cout << std::endl;
    std::cout << "  With corner turning (column-major B):" << std::endl;
    std::cout << "    Thread tx loads B at address differing by 1" << std::endl;
    std::cout << "    from its neighbor -> COALESCED (stride-1)" << std::endl;
    std::cout << std::endl;
    std::cout << "  Shared memory B_s stores the TRANSPOSE of the B tile." << std::endl;
    std::cout << "  Dot product access B_s[k][tx] still yields correct" << std::endl;
    std::cout << "  elements because both indices swap symmetrically." << std::endl;
    std::cout << "============================================================" << std::endl;

    // -----------------------------------------------------------------------
    // Cleanup
    // -----------------------------------------------------------------------
    CHECK_CUDA(cudaFree((void*)d_A));
    CHECK_CUDA(cudaFree((void*)d_B_col));
    CHECK_CUDA(cudaFree(d_P));
    delete[] h_A;
    delete[] h_B_row;
    delete[] h_B_col;
    delete[] h_P_gpu;
    delete[] h_P_cpu;

    return 0;
}
