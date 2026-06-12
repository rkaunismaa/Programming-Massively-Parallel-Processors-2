/*
 * ch18_dcs_gather.cu — Direct Coulomb Summation: Gather Approach (Fig 18.6)
 *
 * From PMPP 4th Ed, Chapter 18: Electrostatic Potential Map
 * Fig 18.6: DCS kernel using gather approach with constant memory caching.
 *
 * Each thread computes the electrostatic potential at one grid point by
 * summing contributions from all atoms. Atom data is chunked and placed
 * in __constant__ memory for broadcast access. No atomics needed (owner-
 * computes rule: each thread owns its output grid point).
 *
 * Build: nvcc -std=c++17 -arch=sm_61 -O2 -o ch18_dcs_gather ch18_dcs_gather.cu
 * Run:   ./ch18_dcs_gather
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include "../common/cuda_utils.cuh"

#define CHUNK_SIZE 4096  // atoms per chunk (×4 floats each = 16KB in constant mem)

// Constant memory holds one chunk of atom data: x,y,z,charge for each atom
__constant__ float atoms_const[CHUNK_SIZE * 4];

/*
 * Gather kernel (Fig 18.6): one thread per grid point.
 * 2D thread block maps to 2D grid-point space.
 * Each thread accumulates contributions from all atoms in the current chunk.
 * Called repeatedly for each atom chunk; energygrid is accumulated across calls.
 */
__global__ void dcs_gather_kernel(float *energygrid, dim3 grid, float gridspacing,
                                   float z, int numatoms_in_chunk)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= grid.x || j >= grid.y) return;

    int atomarrdim = numatoms_in_chunk * 4;
    int k = (int)(z / gridspacing);

    float y = gridspacing * (float)j;
    float x = gridspacing * (float)i;
    float energy = 0.0f;

    // Sum contributions from all atoms in this chunk
    for (int n = 0; n < atomarrdim; n += 4) {
        float dx = x - atoms_const[n];
        float dy = y - atoms_const[n + 1];
        float dz = z - atoms_const[n + 2];
        energy += atoms_const[n + 3] / sqrtf(dx*dx + dy*dy + dz*dz);
    }

    // Accumulate into global energy grid (multiple chunks sum into same grid point)
    energygrid[grid.x * grid.y * k + grid.x * j + i] += energy;
}

/*
 * CPU reference: Direct Coulomb Summation (matches Fig 18.6 semantics).
 * Computes energygrid for one 2D slice at height z.
 */
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
    CHECK_CUDA(cudaSetDevice(1));  // GTX 1050

    // Grid parameters
    const int GRID_X = 64;
    const int GRID_Y = 64;
    const float GRID_SPACING = 0.5f;
    const float SLICE_Z = 1.5f;  // z-coordinate of this 2D slice
    const int NUM_ATOMS = 2000;

    dim3 grid_dim;
    grid_dim.x = GRID_X;
    grid_dim.y = GRID_Y;
    grid_dim.z = 1;

    // Allocate energy grid (one full 3D grid but we only use one slice)
    int total_grid_points = GRID_X * GRID_Y * 4;  // 4 slices worth
    int slice_offset = GRID_X * GRID_Y * (int)(SLICE_Z / GRID_SPACING);
    size_t grid_bytes = total_grid_points * sizeof(float);

    float *h_energygrid = (float *)calloc(total_grid_points, sizeof(float));
    float *h_reference = (float *)calloc(total_grid_points, sizeof(float));

    // Generate random atom data: x,y,z ∈ [0, GRID_X*GRID_SPACING), charge ∈ [-1, 1]
    srand(42);
    float *h_atoms = (float *)malloc(NUM_ATOMS * 4 * sizeof(float));
    float max_coord = GRID_X * GRID_SPACING;
    for (int i = 0; i < NUM_ATOMS; i++) {
        h_atoms[i*4 + 0] = (float)rand() / RAND_MAX * max_coord;
        h_atoms[i*4 + 1] = (float)rand() / RAND_MAX * max_coord;
        h_atoms[i*4 + 2] = (float)rand() / RAND_MAX * max_coord;
        h_atoms[i*4 + 3] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;  // charge
    }

    // CPU reference
    cpu_dcs_reference(h_reference, GRID_X, GRID_Y, GRID_SPACING,
                      SLICE_Z, h_atoms, NUM_ATOMS);

    // Allocate device memory
    float *d_energygrid;
    CHECK_CUDA(cudaMalloc(&d_energygrid, grid_bytes));
    CHECK_CUDA(cudaMemset(d_energygrid, 0, grid_bytes));

    // Warm-up: process first chunk
    int atoms_processed = 0;
    int chunk_atoms = (NUM_ATOMS < CHUNK_SIZE) ? NUM_ATOMS : CHUNK_SIZE;
    CHECK_CUDA(cudaMemcpyToSymbol(atoms_const, h_atoms,
                                   chunk_atoms * 4 * sizeof(float), 0,
                                   cudaMemcpyHostToDevice));

    dim3 block_dim(16, 16);
    dim3 grid_config((GRID_X + 15) / 16, (GRID_Y + 15) / 16);
    dcs_gather_kernel<<<grid_config, block_dim>>>(d_energygrid, grid_dim,
                                                   GRID_SPACING, SLICE_Z,
                                                   chunk_atoms);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Reset for timed run
    CHECK_CUDA(cudaMemset(d_energygrid, 0, grid_bytes));

    // Process all chunks (timed)
    gpu_timer timer;
    timer.start();

    atoms_processed = 0;
    while (atoms_processed < NUM_ATOMS) {
        int remaining = NUM_ATOMS - atoms_processed;
        chunk_atoms = (remaining > CHUNK_SIZE) ? CHUNK_SIZE : remaining;

        CHECK_CUDA(cudaMemcpyToSymbol(atoms_const,
                                       &h_atoms[atoms_processed * 4],
                                       chunk_atoms * 4 * sizeof(float), 0,
                                       cudaMemcpyHostToDevice));

        dcs_gather_kernel<<<grid_config, block_dim>>>(d_energygrid, grid_dim,
                                                       GRID_SPACING, SLICE_Z,
                                                       chunk_atoms);
        atoms_processed += CHUNK_SIZE;
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    timer.stop();
    float elapsed_ms = timer.elapsed_ms();

    // Copy back
    CHECK_CUDA(cudaMemcpy(h_energygrid, d_energygrid, grid_bytes,
                           cudaMemcpyDeviceToHost));

    // Validate: only the computed slice
    int n_check = GRID_X * GRID_Y;
    float *gpu_slice = &h_energygrid[slice_offset];
    float *ref_slice = &h_reference[slice_offset];

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

    printf("ch18_dcs_gather  | GRID %dx%d | ATOMS %d | CHUNK %d | %s | %.3f ms\n",
           GRID_X, GRID_Y, NUM_ATOMS, CHUNK_SIZE,
           pass ? "PASS" : "FAIL", elapsed_ms);

    // Cleanup
    CHECK_CUDA(cudaFree(d_energygrid));
    free(h_energygrid);
    free(h_reference);
    free(h_atoms);

    return pass ? 0 : 1;
}
