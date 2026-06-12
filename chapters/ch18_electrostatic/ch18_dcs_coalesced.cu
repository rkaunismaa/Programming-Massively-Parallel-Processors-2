/*
 * ch18_dcs_coalesced.cu — Direct Coulomb Summation: Coarsening + Coalescing (Fig 18.10)
 *
 * From PMPP 4th Ed, Chapter 18: Electrostatic Potential Map
 * Fig 18.10: Thread coarsening WITH memory coalescing.
 *
 * Key difference from Fig 18.8: grid points assigned to each thread are
 * INTERLEAVED, not consecutive. Adjacent threads access adjacent locations
 * in each write statement → fully coalesced global memory writes.
 *
 * Grid point offset: stride = blockDim.x * gridspacing (not gridspacing)
 * Write offset:       stride = blockDim.x (not 1)
 *
 * This maps to Fig 18.9's assignment strategy: assign blockDim.x consecutive
 * grid points to threads, then next blockDim.x to same threads, etc.
 *
 * Build: nvcc -std=c++17 -arch=sm_61 -O2 -o ch18_dcs_coalesced ch18_dcs_coalesced.cu
 * Run:   ./ch18_dcs_coalesced
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include "../common/cuda_utils.cuh"

#define CHUNK_SIZE 4096
#define COARSEN_FACTOR 4

__constant__ float atoms_const[CHUNK_SIZE * 4];

/*
 * Coarsened + coalesced gather kernel (Fig 18.10).
 * Each thread handles 4 grid points in x, interleaved by blockDim.x.
 * Thread 0 → grid points [0, BDx, 2*BDx, 3*BDx]
 * Thread 1 → grid points [1, BDx+1, 2*BDx+1, 3*BDx+1]
 * This ensures write coalescing: adjacent threads write adjacent locations.
 */
__global__ void dcs_coalesced_kernel(float *energygrid, dim3 grid, float gridspacing,
                                      float z, int numatoms_in_chunk)
{
    int i = blockIdx.x * blockDim.x * COARSEN_FACTOR + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= grid.x || j >= grid.y) return;

    int atomarrdim = numatoms_in_chunk * 4;
    int k = (int)(z / gridspacing);

    float y = gridspacing * (float)j;
    float x = gridspacing * (float)i;

    float energy0 = 0.0f;
    float energy1 = 0.0f;
    float energy2 = 0.0f;
    float energy3 = 0.0f;

    // Interleaved stride: blockDim.x * gridspacing (for physical distance)
    float stride = (float)blockDim.x * gridspacing;

    for (int n = 0; n < atomarrdim; n += 4) {
        float dx0 = x - atoms_const[n];
        float dx1 = dx0 + stride;
        float dx2 = dx0 + 2.0f * stride;
        float dx3 = dx0 + 3.0f * stride;
        float dy  = y - atoms_const[n + 1];
        float dz  = z - atoms_const[n + 2];
        float dysqdzsq = dy*dy + dz*dz;
        float charge = atoms_const[n + 3];

        energy0 += charge / sqrtf(dx0*dx0 + dysqdzsq);
        energy1 += charge / sqrtf(dx1*dx1 + dysqdzsq);
        energy2 += charge / sqrtf(dx2*dx2 + dysqdzsq);
        energy3 += charge / sqrtf(dx3*dx3 + dysqdzsq);
    }

    // Coalesced writes: interleaved by blockDim.x
    int base = grid.x * grid.y * k + grid.x * j + i;
    int bdx = (int)blockDim.x;
    energygrid[base]            += energy0;
    energygrid[base + bdx]      += energy1;
    energygrid[base + 2 * bdx]  += energy2;
    energygrid[base + 3 * bdx]  += energy3;
}

void cpu_dcs_reference(float *energygrid, int gx, int gy, float gridspacing,
                       float z, const float *atoms, int numatoms)
{
    int atomarrdim = numatoms * 4;
    int k = (int)(z / gridspacing);

    for (int j = 0; j < gy; j++) {
        float y = gridspacing * (float)j;
        for (int i = 0; i < gx; i++) {
            float x = gridspacing * (float)i;
            float energy = 0.0f;
            for (int n = 0; n < atomarrdim; n += 4) {
                float dx = x - atoms[n];
                float dy = y - atoms[n + 1];
                float dz = z - atoms[n + 2];
                energy += atoms[n + 3] / sqrtf(dx*dx + dy*dy + dz*dz);
            }
            energygrid[gx * gy * k + gx * j + i] = energy;
        }
    }
}

int main()
{
    CHECK_CUDA(cudaSetDevice(1));

    const int GRID_X = 64;
    const int GRID_Y = 64;
    const float GRID_SPACING = 0.5f;
    const float SLICE_Z = 1.5f;
    const int NUM_ATOMS = 2000;

    dim3 grid_dim;
    grid_dim.x = GRID_X;
    grid_dim.y = GRID_Y;

    int total_grid_points = GRID_X * GRID_Y * 4;
    int slice_offset = GRID_X * GRID_Y * (int)(SLICE_Z / GRID_SPACING);
    size_t grid_bytes = total_grid_points * sizeof(float);

    float *h_energygrid = (float *)calloc(total_grid_points, sizeof(float));
    float *h_reference = (float *)calloc(total_grid_points, sizeof(float));

    srand(42);
    float *h_atoms = (float *)malloc(NUM_ATOMS * 4 * sizeof(float));
    float max_coord = GRID_X * GRID_SPACING;
    for (int i = 0; i < NUM_ATOMS; i++) {
        h_atoms[i*4 + 0] = (float)rand() / RAND_MAX * max_coord;
        h_atoms[i*4 + 1] = (float)rand() / RAND_MAX * max_coord;
        h_atoms[i*4 + 2] = (float)rand() / RAND_MAX * max_coord;
        h_atoms[i*4 + 3] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }

    cpu_dcs_reference(h_reference, GRID_X, GRID_Y, GRID_SPACING,
                      SLICE_Z, h_atoms, NUM_ATOMS);

    float *d_energygrid;
    CHECK_CUDA(cudaMalloc(&d_energygrid, grid_bytes));
    CHECK_CUDA(cudaMemset(d_energygrid, 0, grid_bytes));

    // Warm-up
    int chunk_atoms = (NUM_ATOMS < CHUNK_SIZE) ? NUM_ATOMS : CHUNK_SIZE;
    CHECK_CUDA(cudaMemcpyToSymbol(atoms_const, h_atoms,
                                   chunk_atoms * 4 * sizeof(float), 0,
                                   cudaMemcpyHostToDevice));

    dim3 block_dim(16, 16);
    int grid_x = (GRID_X + block_dim.x * COARSEN_FACTOR - 1) / (block_dim.x * COARSEN_FACTOR);
    int grid_y = (GRID_Y + block_dim.y - 1) / block_dim.y;
    dim3 grid_config(grid_x, grid_y);
    dcs_coalesced_kernel<<<grid_config, block_dim>>>(d_energygrid, grid_dim,
                                                      GRID_SPACING, SLICE_Z,
                                                      chunk_atoms);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Reset for timed run
    CHECK_CUDA(cudaMemset(d_energygrid, 0, grid_bytes));

    gpu_timer timer;
    timer.start();

    int atoms_processed = 0;
    while (atoms_processed < NUM_ATOMS) {
        int remaining = NUM_ATOMS - atoms_processed;
        chunk_atoms = (remaining > CHUNK_SIZE) ? CHUNK_SIZE : remaining;

        CHECK_CUDA(cudaMemcpyToSymbol(atoms_const,
                                       &h_atoms[atoms_processed * 4],
                                       chunk_atoms * 4 * sizeof(float), 0,
                                       cudaMemcpyHostToDevice));

        dcs_coalesced_kernel<<<grid_config, block_dim>>>(d_energygrid, grid_dim,
                                                          GRID_SPACING, SLICE_Z,
                                                          chunk_atoms);
        atoms_processed += CHUNK_SIZE;
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    timer.stop();
    float elapsed_ms = timer.elapsed_ms();

    CHECK_CUDA(cudaMemcpy(h_energygrid, d_energygrid, grid_bytes,
                           cudaMemcpyDeviceToHost));

    float *gpu_slice = &h_energygrid[slice_offset];
    float *ref_slice = &h_reference[slice_offset];
    int n_check = GRID_X * GRID_Y;

    bool pass = true;
    int fail_count = 0;
    for (int idx = 0; idx < n_check && pass; idx++) {
        float diff = fabsf(gpu_slice[idx] - ref_slice[idx]);
        float denom = fmaxf(fabsf(ref_slice[idx]), 1e-10f);
        if (diff / denom > 1e-4f && diff > 1e-6f) {
            printf("FAIL at [%d,%d]: gpu=%.6e ref=%.6e diff=%.2e\n",
                   idx % GRID_X, idx / GRID_X, gpu_slice[idx], ref_slice[idx], diff);
            fail_count++;
            if (fail_count >= 5) { pass = false; break; }
        }
    }

    printf("ch18_dcs_coalesced | GRID %dx%d | ATOMS %d | COARSE %d | %s | %.3f ms\n",
           GRID_X, GRID_Y, NUM_ATOMS, COARSEN_FACTOR,
           pass ? "PASS" : "FAIL", elapsed_ms);

    CHECK_CUDA(cudaFree(d_energygrid));
    free(h_energygrid);
    free(h_reference);
    free(h_atoms);

    return pass ? 0 : 1;
}
