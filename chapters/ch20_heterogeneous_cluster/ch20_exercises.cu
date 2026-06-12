/*
 * ch20_exercises.cu — Chapter 20 Exercises
 *
 * Exercise 1: Stencil grid partitioning calculations
 * Exercise 2: MPI element size
 * Exercise 3: MPI true/false
 * Exercise 4: CUDA-aware MPI code (conceptual — requires MPI to compile)
 *
 * Build: nvcc -std=c++17 -arch=sm_61 -O2 -o ch20_exercises ch20_exercises.cu
 * Run:   ./ch20_exercises
 */

#include <stdio.h>

int main()
{
    printf("=== Chapter 20 Exercises ===\n\n");

    // ============================================
    // Exercise 1: Stencil grid partitioning (25-point stencil, 4-slice halo)
    // ============================================
    printf("Exercise 1:\n");
    printf("  Grid: 64 x 64 x 2048, 17 MPI ranks (16 compute + 1 server)\n");
    printf("  Stencil: 25-point (radius=2, so halo=2 slices each side)\n\n");

    int dimx = 64, dimy = 64, dimz = 2048;
    int halo = 2;  // 25-point stencil has radius 2
    int num_compute = 16;

    // (a) Output grid points per compute process
    // Full grid has dimz slices. Each process gets dimz/16 slices internally,
    // but the stencil needs halo on each side.
    // The output region per process = dimz/num_compute = 2048/16 = 128
    int slices_per_proc = dimz / num_compute;
    long long output_per_proc = (long long)dimx * dimy * slices_per_proc;
    printf("  (a) Output grid points per compute process:\n");
    printf("      %lld (= %d x %d x %d)\n\n", output_per_proc, dimx, dimy, slices_per_proc);

    // (b) Halo grid points needed
    // Each slice is dimx × dimy points. Halo = 2 on each side, so 2 halo slices per side.
    long long halo_per_side = (long long)dimx * dimy * halo;
    printf("  (b) Halo grid points needed:\n");
    printf("      (i) Internal compute process: %lld (= 2 sides x %d halo x %dx%d)\n",
           (long long)dimx * dimy * halo * 2, halo, dimx, dimy);
    printf("      (ii) Edge compute process: %lld (= 1 side x %d halo x %dx%d)\n",
           (long long)dimx * dimy * halo, halo, dimx, dimy);
    printf("          (processes 0 and 15 have only one neighbor)\n\n");

    // (c) Boundary points in stage 1 (Fig 20.12)
    // Stage 1 computes boundary slices that are needed by neighbors.
    // To correctly compute 2 output boundary slices via 25-pt stencil (radius=2),
    // we need input data from 2+2=4 slices inward, hence 4 boundary slices computed.
    printf("  (c) Boundary grid points in stage 1:\n");
    printf("      (i) Internal compute process: %lld (= 2 sides x 4 slices x %dx%d)\n",
           (long long)dimx * dimy * 4 * 2, dimx, dimy);
    printf("      (ii) Edge compute process: %lld (= 1 side x 4 slices x %dx%d)\n",
           (long long)dimx * dimy * 4, dimx, dimy);
    printf("          (edge process only sends to one neighbor)\n\n");

    // (d) Internal points in stage 2
    // Internal = total slices - 2*h_boundary - 2*h_halo
    // Each process gets slices_per_proc total slices in its partition.
    // But it also has 2 halo slices on each side = 4 halo slices
    // And 4 boundary slices on each side for stage 1 = need 8 slices of input
    // The internal region = dimz/16 - 2*4 = 128 - 8
    int internal_slices = slices_per_proc - 2 * 4;  // 4 boundary slices per side
    printf("  (d) Internal grid points in stage 2:\n");
    printf("      (i) Internal compute process: %lld (= %d slices x %dx%d)\n",
           (long long)dimx * dimy * internal_slices, internal_slices, dimx, dimy);
    printf("      (ii) Edge compute process: %lld (same — internal region unaffected by edge)\n\n",
           (long long)dimx * dimy * internal_slices);

    // (e) Bytes sent in stage 2
    // Each process sends halo slices to neighbors.
    // 2 halo slices per side, each dimx × dimy × sizeof(float) = 4 bytes
    long long halo_bytes_per_side = (long long)dimx * dimy * halo * sizeof(float);
    printf("  (e) Bytes sent in stage 2:\n");
    printf("      (i) Internal compute process: %lld bytes (= 2 sides x %d halo x %dx%d x 4B)\n",
           halo_bytes_per_side * 2, halo, dimx, dimy);
    printf("      (ii) Edge compute process: %lld bytes (= 1 side x %d halo x %dx%d x 4B)\n",
           halo_bytes_per_side, halo, dimx, dimy);

    // ============================================
    // Exercise 2: MPI element size
    // ============================================
    printf("\nExercise 2:\n");
    printf("  MPI_Send(ptr_a, 1000, MPI_FLOAT, 2000, 4, MPI_COMM_WORLD)\n");
    printf("  Sends 4000 bytes for 1000 elements.\n");
    printf("  4000 / 1000 = 4 bytes per element.\n");
    printf("  Answer: (c) 4 bytes (MPI_FLOAT = 4 bytes)\n");

    // ============================================
    // Exercise 3: MPI true/false
    // ============================================
    printf("\nExercise 3:\n");
    printf("  (a) MPI_Send() is blocking by default — FALSE.\n");
    printf("      MPI_Send() may or may not block depending on implementation.\n");
    printf("      The standard allows the implementation to buffer or block.\n");
    printf("  (b) MPI_Recv() is blocking by default — TRUE.\n");
    printf("      MPI_Recv() blocks until the message is received.\n");
    printf("  (c) MPI messages must be at least 128 bytes — FALSE.\n");
    printf("      MPI messages can be any size, including zero bytes.\n");
    printf("  (d) MPI processes can access the same variable through\n");
    printf("      shared memory — FALSE.\n");
    printf("      MPI processes have separate address spaces.\n");
    printf("      Communication is via explicit message passing.\n");
    printf("  Answer: (b) is the only true statement.\n");

    // ============================================
    // Exercise 4: CUDA-aware MPI code modification
    // ============================================
    printf("\nExercise 4: CUDA-aware MPI modifications\n");
    printf("  See ch20_ex4_cuda_aware_mpi.cu for the code comparison.\n");
    printf("  Key changes:\n");
    printf("  1. Remove cudaHostAlloc() for halo bounce buffers\n");
    printf("  2. Remove cudaMemcpyAsync() D2H before MPI_Sendrecv\n");
    printf("  3. Remove cudaMemcpyAsync() H2D after MPI_Sendrecv\n");
    printf("  4. MPI_Sendrecv now uses device pointers directly\n");
    printf("  5. MPI_Send at data collection uses device pointer\n");

    return 0;
}
