// Book:      Programming Massively Parallel Processors (Kirk, Hwu & El Hajj, 4th ed.)
// Chapter:   3 — Multidimensional Grids and Data
// Reference: Figure 3.8
// Concept:   Image blur kernel — each thread averages a patch of neighboring pixels
// Key insight: nested loops over a BLUR_SIZE patch; boundary handling via conditional
//              checks inside the loop so edge pixels use only valid neighbors
// Hardware:  RTX 4090, sm_89 (Ada Lovelace)
// Compile:   nvcc -std=c++17 -arch=sm_89 -O2 ch03_image_blur_fig_3_8.cu -o image_blur

#include "../common/cuda_utils.cuh"
#include <cstring>

// ============================================================
// Kernel: blurKernel (Figure 3.8)
// Each thread computes one output pixel as the average of a
// (2*BLUR_SIZE+1) x (2*BLUR_SIZE+1) patch centered on that pixel
// ============================================================
#define BLUR_SIZE 1

__global__
void blurKernel(const unsigned char* Pin_d, unsigned char* Pout_d,
                int width, int height) {
    // Compute 2D thread coordinates
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // Boundary check
    if (col < width && row < height) {
        float pixVal = 0.0f;
        int pixels = 0;

        // Iterate over the patch centered at (row, col)
        for (int blurRow = -BLUR_SIZE; blurRow <= BLUR_SIZE; blurRow++) {
            for (int blurCol = -BLUR_SIZE; blurCol <= BLUR_SIZE; blurCol++) {
                int curRow = row + blurRow;
                int curCol = col + blurCol;

                // Only accumulate valid pixels (boundary handling)
                if (curRow >= 0 && curRow < height &&
                    curCol >= 0 && curCol < width) {
                    pixVal += (float)Pin_d[curRow * width + curCol];
                    pixels++;
                }
            }
        }

        // Write the average
        Pout_d[row * width + col] = (unsigned char)(pixVal / pixels);
    }
}

// ============================================================
// Host wrapper
// ============================================================
void imageBlur(unsigned char* Pin_h, unsigned char* Pout_h,
               int width, int height) {
    int size = width * height * sizeof(unsigned char);

    unsigned char *Pin_d, *Pout_d;
    CHECK_CUDA(cudaMalloc((void**)&Pin_d, size));
    CHECK_CUDA(cudaMalloc((void**)&Pout_d, size));

    CHECK_CUDA(cudaMemcpy(Pin_d, Pin_h, size, cudaMemcpyHostToDevice));

    // 2D block and grid configuration
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);

    blurKernel<<<grid, block>>>(Pin_d, Pout_d, width, height);
    CHECK_CUDA(cudaGetLastError());

    CHECK_CUDA(cudaMemcpy(Pout_h, Pout_d, size, cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(Pin_d));
    CHECK_CUDA(cudaFree(Pout_d));
}

int main() {
    print_device_info();

    int width = 1920;
    int height = 1080;
    int num_pixels = width * height;

    printf("\nImage size: %d x %d = %d pixels\n", width, height, num_pixels);
    printf("Blur patch:  %dx%d\n", 2 * BLUR_SIZE + 1, 2 * BLUR_SIZE + 1);

    // Allocate host memory
    unsigned char* Pin_h = new unsigned char[num_pixels];
    unsigned char* Pout_h = new unsigned char[num_pixels];
    unsigned char* expected = new unsigned char[num_pixels];

    // Initialize with test data (gradient pattern for visual verifiability)
    for (int i = 0; i < num_pixels; i++) {
        int row = i / width;
        int col = i % width;
        Pin_h[i] = (unsigned char)((row + col) % 256);
    }

    // CPU reference — identical blur logic
    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            float pixVal = 0.0f;
            int pixels = 0;
            for (int blurRow = -BLUR_SIZE; blurRow <= BLUR_SIZE; blurRow++) {
                for (int blurCol = -BLUR_SIZE; blurCol <= BLUR_SIZE; blurCol++) {
                    int curRow = row + blurRow;
                    int curCol = col + blurCol;
                    if (curRow >= 0 && curRow < height &&
                        curCol >= 0 && curCol < width) {
                        pixVal += (float)Pin_h[curRow * width + curCol];
                        pixels++;
                    }
                }
            }
            expected[row * width + col] = (unsigned char)(pixVal / pixels);
        }
    }

    // GPU execution with timing
    gpu_timer timer;
    timer.start();
    imageBlur(Pin_h, Pout_h, width, height);
    timer.stop();
    float gpu_ms = timer.elapsed_ms();

    // Validate — exact match expected (no floating point)
    bool pass = true;
    int mismatches = 0;
    for (int i = 0; i < num_pixels; i++) {
        if (Pout_h[i] != expected[i]) {
            mismatches++;
            if (mismatches <= 5) {
                printf("  Mismatch at pixel %d: expected %d, got %d\n",
                       i, expected[i], Pout_h[i]);
            }
            pass = false;
        }
    }
    if (mismatches > 5) {
        printf("  ... and %d more mismatches\n", mismatches - 5);
    }

    printf("\nValidation: %s\n", pass ? "PASS" : "FAIL");
    printf("GPU time:   %.3f ms\n", gpu_ms);

    // Effective throughput
    float total_mb = (float)num_pixels * 2 / (1024.0f * 1024.0f); // read + write
    float bandwidth = total_mb / gpu_ms; // MB/s
    printf("Throughput: %.1f MB/s\n", bandwidth);

    delete[] Pin_h;
    delete[] Pout_h;
    delete[] expected;

    return pass ? 0 : 1;
}
