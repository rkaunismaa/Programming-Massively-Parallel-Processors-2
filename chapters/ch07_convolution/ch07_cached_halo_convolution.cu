/*
 * =============================================================================
 *  Chapter 7: Convolution — Cached Halo Tiles (Fig 7.15)
 * =============================================================================
 *  Book:      Programming Massively Parallel Processors (Kirk, Hwu & El Hajj,
 *             4th Edition, Morgan Kaufmann 2023)
 *  Figure:    7.15 — Tiled convolution using caches for halo cells
 *  Purpose:   Input tile loaded without explicit halo padding — each
 *             TILE_DIM thread reads exactly d_N[row*width+col] into N_s[ty][tx].
 *             Convolution loop checks the shared-memory bounds; out-of-bounds
 *             neighbours come from global memory (L2 cached from neighbour tile).
 * =============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include "../common/cuda_utils.cuh"

#define FILTER_RADIUS 2              // Book Fig 7.15 -- square 5x5 filter
#define TILE_DIM      32             // Block dim == N_s tile dimension
#define FSZ           (2*FILTER_RADIUS+1)
#define OUT_TILE_DIM  (TILE_DIM - 2*FILTER_RADIUS)// 28, interior output per block

// Figure 7.9 / 7.15: constant memory for the filter
__constant__ float F_c[FSZ][FSZ];

/* Figure 7.15 — Tiled convolution kernel with cached halos */
__global__ void convolution_cached_halo_kernel(
    float *N, float *P, int width, int height) {

    // Same thread → global coordinate mapping as Fig 7.12
    int col = blockIdx.x * OUT_TILE_DIM + threadIdx.x - FILTER_RADIUS;
    int row = blockIdx.y * OUT_TILE_DIM + threadIdx.y - FILTER_RADIUS;

    /* -- Load input tile into shared memory (zero-padded halos) -- */
    __shared__ float N_s[TILE_DIM][TILE_DIM];
    if (row >= 0 && row < height && col >= 0 && col < width) {
        N_s[threadIdx.y][threadIdx.x] = N[row * width + col];
    } else {
        N_s[threadIdx.y][threadIdx.x] = 0.0f;   /* ghost cell → 0 */
    }
    __syncthreads();

    /* -- Compute interior output elements -- */
    int tileCol = threadIdx.x - FILTER_RADIUS;
    int tileRow = threadIdx.y - FILTER_RADIUS;

    if (col >= 0 && col < width && row >= 0 && row < height
        && tileCol >= 0 && tileCol < OUT_TILE_DIM
        && tileRow >= 0 && tileRow < OUT_TILE_DIM) {

        float Pvalue = 0.0f;
        for (int fRow = 0; fRow < FSZ; fRow++) {
            for (int fCol = 0; fCol < FSZ; fCol++) {
                
                /* ---- Figure 7.15 logic: try shared-mem first, fall back to global ------ */
                int s_row = tileRow + fRow;   // [0..TILE_DIM-1] when within interior margin
                int s_col = tileCol + fCol;

                if (s_row >= 0 && s_row < TILE_DIM && s_col >= 0 && s_col < TILE_DIM) {
                    /* Interior neighbour already in shared memory */
                    Pvalue += F_c[fRow][fCol] * N_s[s_row][s_col];
                } else {
                    /* Halo neighbour → read from global memory (L2 cache expected hit) */
                    int gRow = row + fRow - FILTER_RADIUS;    // input pixel's global row
                    int gCol = col + fCol - FILTER_RADIUS;    // input pixel's global col

                    if (gRow >= 0 && gRow < height && gCol >= 0 && gCol < width) {
                        Pvalue += F_c[fRow][fCol] * N[gRow * width + gCol];
                    } else {
                        /* Boundary of image — treat as zero */
                    }
                }
            }
        }
        P[row * width + col] = Pvalue;
    }
}

/* CPU reference */
void cpu_convolve(const float* N, const float* F, float* P,
                  int r, int fsz, int width, int height) {
    for (int row = 0; row < height; row++)
        for (int col = 0; col < width; col++) {
            float val = 0.0f;
            for (int fr = 0; fr < fsz; fr++)
                for (int fc = 0; fc < fsz; fc++) {
                    int ir = row - r + fr, ic = col - r + fc;
                    if (ir >= 0 && ir < height && ic >= 0 && ic < width)
                        val += F[fr*fsz+fc] * N[ir*width+ic];
                }
            P[row*width+col] = val;
        }
}

int main() {
    cudaSetDevice(1);
    print_device_info(1);

    int r = FILTER_RADIUS, fsz = FSZ;
    int width  = 1024, height = 1024, n = width*height;

    printf("\n=============================================================\n");
    printf("Chapter 7: Cached-Halo Tiled Convolution (Fig 7.15)\n");
    printf("Tile %d x %d, Out tile %d x %d\n", TILE_DIM,TILE_DIM,OUT_TILE_DIM,OUT_TILE_DIM);

    float *N_h=(float*)malloc(n*sizeof(float));
    float *F_h=(float*)calloc(fsz*fsz,sizeof(float));
    float *P_h=(float*)malloc(n*sizeof(float));
    float *C_h=(float*)malloc(n*sizeof(float));

    for (int i=0;i<n;i++) N_h[i]=fmodf(i*0.01f,10.f);
    for (int fy=0;fy<fsz;fy++)
        for (int fx=0;fx<fsz;fx++){ float dx=fx-r,dy=fy-r;
            F_h[fy*fsz+fx]=expf(-(dx*dx+dy*dy)/2.f); }

    float *dN,*dP;
    CHECK_CUDA(cudaMalloc(&dN,n*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dP,n*sizeof(float)));
    CHECK_CUDA(cudaMemcpy(dN,N_h,n*sizeof(float),cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpyToSymbol(F_c,F_h,fsz*fsz*sizeof(float)));

    dim3 block(TILE_DIM,TILE_DIM);
    dim3 grid((width+OUT_TILE_DIM-1)/OUT_TILE_DIM,
              (height+OUT_TILE_DIM-1)/OUT_TILE_DIM);
    printf("Grid %dx%d (%d blocks), Block %dx%d\n",
           grid.x,grid.y,grid.x*grid.y,block.x,block.y);

    /* Warmup */
    convolution_cached_halo_kernel<<<grid,block>>>(dN,dP,width,height);
    CHECK_CUDA(cudaDeviceSynchronize());

    /* Timed run */
    gpu_timer timer;
    timer.start();
    convolution_cached_halo_kernel<<<grid,block>>>(dN,dP,width,height);
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.stop();

    CHECK_CUDA(cudaMemcpy(P_h,dP,n*sizeof(float),cudaMemcpyDeviceToHost));
    cpu_convolve(N_h,F_h,C_h,r,fsz,width,height);
    bool passed = cpu_allclose(P_h,C_h,n,1e-2f);

    float elapsed = timer.elapsed_ms();
    printf("\nTime: %.2f ms\n", elapsed);
    float gflops = (double)n * fsz*fsz * 2 / (elapsed*1e6);
    printf("GFLOPS: %.2f\n",gflops);
    printf("%s\n", passed?"Validation: PASSED":"Validation: FAILED");

    CHECK_CUDA(cudaFree(dN)); CHECK_CUDA(cudaFree(dP));
    free(N_h);free(F_h);free(P_h);free(C_h);
    return 0;
}
