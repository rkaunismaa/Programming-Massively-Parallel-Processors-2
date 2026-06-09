// Book:      Programming Massively Parallel Processors (Kirk, Hwu & El Hajj, 4th ed.)
// Chapter:   3 — Multidimensional Grids and Data
// Reference: Figure 3.4
// Concept:   Color-to-grayscale conversion using 2D thread mapping
// Key insight: 2D grids map naturally to 2D data (images); row-major linearization
//              converts (row, col) to 1D index: row * width + col
// Hardware:  GTX 1050, sm_61 (Pascal)
// Compile:   nvcc -std=c++17 -arch=sm_61 -O2 ch03_color_to_grayscale_fig_3_4.cu -o color_to_grayscale

#include "../common/cuda_utils.cuh"
#include <cstring>

// ============================================================
// Kernel: colorToGrayscaleConversion (Figure 3.4)
// Each thread computes one output pixel from RGB input
// Luminance = 0.21*R + 0.72*G + 0.07*B
// ============================================================
__global__
void colorToGrayscaleConversion(const float* Pin_d, float* Pout_d,
                                int width, int height) {
    // Compute 2D thread coordinates
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // Boundary check — threads outside image do nothing
    if (col < width && row < height) {
        // Row-major linearization: index = row * width + col
        int idx = row * width + col;

        // Each pixel has 3 channels (R, G, B) stored consecutively
        float r = Pin_d[3 * idx];
        float g = Pin_d[3 * idx + 1];
        float b = Pin_d[3 * idx + 2];

        // Luminance formula (ITU-R BT.709 weights)
        Pout_d[idx] = 0.21f * r + 0.72f * g + 0.07f * b;
    }
}

// ============================================================
// Host wrapper
// ============================================================
void colorToGrayscale(float* Pin_h, float* Pout_h, int width, int height) {
    int num_pixels = width * height;
    int size_in = num_pixels * 3 * sizeof(float); // RGB input
    int size_out = num_pixels * sizeof(float);    // Grayscale output

    float *Pin_d, *Pout_d;
    CHECK_CUDA(cudaMalloc((void**)&Pin_d, size_in));
    CHECK_CUDA(cudaMalloc((void**)&Pout_d, size_out));

    CHECK_CUDA(cudaMemcpy(Pin_d, Pin_h, size_in, cudaMemcpyHostToDevice));

    // 2D block and grid configuration
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);

    colorToGrayscaleConversion<<<grid, block>>>(Pin_d, Pout_d, width, height);
    CHECK_CUDA(cudaGetLastError());

    CHECK_CUDA(cudaMemcpy(Pout_h, Pout_d, size_out, cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(Pin_d));
    CHECK_CUDA(cudaFree(Pout_d));
}

int main() {
    print_device_info();

    int width = 1920;
    int height = 1080;
    int num_pixels = width * height;

    printf("\nImage size: %d x %d = %d pixels\n", width, height, num_pixels);

    // Allocate host memory
    float* Pin_h = new float[num_pixels * 3];
    float* Pout_h = new float[num_pixels];
    float* expected = new float[num_pixels];

    // Initialize with test data (deterministic pattern)
    for (int i = 0; i < num_pixels; i++) {
        Pin_h[3 * i]     = (float)((i * 7) % 256) / 255.0f;  // R
        Pin_h[3 * i + 1] = (float)((i * 13) % 256) / 255.0f; // G
        Pin_h[3 * i + 2] = (float)((i * 19) % 256) / 255.0f; // B
    }

    // CPU reference
    for (int i = 0; i < num_pixels; i++) {
        expected[i] = 0.21f * Pin_h[3 * i] +
                      0.72f * Pin_h[3 * i + 1] +
                      0.07f * Pin_h[3 * i + 2];
    }

    // GPU execution with timing
    gpu_timer timer;
    timer.start();
    colorToGrayscale(Pin_h, Pout_h, width, height);
    timer.stop();
    float gpu_ms = timer.elapsed_ms();

    // Validate
    bool pass = cpu_allclose(Pout_h, expected, num_pixels, 1e-5f);
    printf("\nValidation: %s\n", pass ? "PASS" : "FAIL");
    printf("GPU time:   %.3f ms\n", gpu_ms);

    // Effective throughput
    float total_bytes = (float)(num_pixels * 3 * sizeof(float) + num_pixels * sizeof(float));
    float bandwidth = (total_bytes * 1e-6f) / gpu_ms; // MB/s
    printf("Throughput: %.1f MB/s\n", bandwidth);

    delete[] Pin_h;
    delete[] Pout_h;
    delete[] expected;

    return pass ? 0 : 1;
}
