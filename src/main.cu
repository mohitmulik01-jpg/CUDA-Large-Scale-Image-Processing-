
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

#define CUDA_CHECK(call) do { \
    cudaError_t error = (call); \
    if (error != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(error) \
                  << " at line " << __LINE__ << std::endl; \
        std::exit(EXIT_FAILURE); \
    } \
} while (0)

struct Image {
    int width;
    int height;
    std::vector<unsigned char> pixels;
};

std::string readToken(std::istream& input) {
    std::string token;
    char ch;

    while (input.get(ch)) {
        if (ch == '#') {
            input.ignore(10000, '\n');
            continue;
        }
        if (!std::isspace(static_cast<unsigned char>(ch))) {
            token += ch;
            break;
        }
    }

    while (input.get(ch)) {
        if (std::isspace(static_cast<unsigned char>(ch))) {
            break;
        }
        token += ch;
    }

    return token;
}

Image loadPPM(const fs::path& path) {
    std::ifstream file(path, std::ios::binary);

    if (!file) {
        throw std::runtime_error("Cannot open " + path.string());
    }

    if (readToken(file) != "P6") {
        throw std::runtime_error("Only binary P6 PPM is supported");
    }

    Image image;
    image.width = std::stoi(readToken(file));
    image.height = std::stoi(readToken(file));

    int maxValue = std::stoi(readToken(file));

    if (image.width <= 0 || image.height <= 0 || maxValue != 255) {
        throw std::runtime_error("Invalid PPM dimensions or color depth");
    }

    size_t bytes = static_cast<size_t>(image.width) *
                   image.height * 3;

    image.pixels.resize(bytes);
    file.read(reinterpret_cast<char*>(image.pixels.data()), bytes);

    if (static_cast<size_t>(file.gcount()) != bytes) {
        throw std::runtime_error("Incomplete image data");
    }

    return image;
}

void savePPM(const fs::path& path,
             const std::vector<unsigned char>& pixels,
             int width, int height) {
    std::ofstream file(path, std::ios::binary);

    if (!file) {
        throw std::runtime_error("Cannot create output image");
    }

    file << "P6\n" << width << " " << height << "\n255\n";

    file.write(reinterpret_cast<const char*>(pixels.data()),
               pixels.size());
}

// Kernel 1: RGB to grayscale conversion
__global__ void grayscaleKernel(
    const unsigned char* input,
    unsigned char* gray,
    int totalPixels) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < totalPixels) {
        int offset = idx * 3;

        float r = input[offset];
        float g = input[offset + 1];
        float b = input[offset + 2];

        gray[idx] = static_cast<unsigned char>(
            0.299f * r + 0.587f * g + 0.114f * b
        );
    }
}

// Kernel 2: 3x3 Gaussian blur
__global__ void gaussianKernel(
    const unsigned char* input,
    unsigned char* output,
    int width, int height) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    const float kernel[9] = {
        1.0f, 2.0f, 1.0f,
        2.0f, 4.0f, 2.0f,
        1.0f, 2.0f, 1.0f
    };

    float sum = 0.0f;
    int k = 0;

    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            int px = min(max(x + dx, 0), width - 1);
            int py = min(max(y + dy, 0), height - 1);

            sum += input[py * width + px] * kernel[k++];
        }
    }

    output[y * width + x] =
        static_cast<unsigned char>(sum / 16.0f);
}

// Kernel 3: Sobel edge detection
__global__ void sobelKernel(
    const unsigned char* input,
    unsigned char* output,
    int width, int height) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    const int gx[9] = {
        -1, 0, 1,
        -2, 0, 2,
        -1, 0, 1
    };

    const int gy[9] = {
        -1, -2, -1,
         0,  0,  0,
         1,  2,  1
    };

    float sx = 0.0f;
    float sy = 0.0f;
    int k = 0;

    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            int px = min(max(x + dx, 0), width - 1);
            int py = min(max(y + dy, 0), height - 1);

            float value = input[py * width + px];

            sx += value * gx[k];
            sy += value * gy[k];
            k++;
        }
    }

    float magnitude = sqrtf(sx * sx + sy * sy);

    output[y * width + x] =
        static_cast<unsigned char>(fminf(magnitude, 255.0f));
}

void processImage(
    const fs::path& inputPath,
    const fs::path& outputPath,
    double& totalGpuMs) {

    Image image = loadPPM(inputPath);

    int width = image.width;
    int height = image.height;
    int totalPixels = width * height;

    size_t rgbBytes = static_cast<size_t>(totalPixels) * 3;
    size_t grayBytes = static_cast<size_t>(totalPixels);

    unsigned char *d_input, *d_gray, *d_blur, *d_output;

    CUDA_CHECK(cudaMalloc(&d_input, rgbBytes));
    CUDA_CHECK(cudaMalloc(&d_gray, grayBytes));
    CUDA_CHECK(cudaMalloc(&d_blur, grayBytes));
    CUDA_CHECK(cudaMalloc(&d_output, grayBytes));

    CUDA_CHECK(cudaMemcpy(
        d_input, image.pixels.data(), rgbBytes,
        cudaMemcpyHostToDevice
    ));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    int threads = 256;
    int blocks = (totalPixels + threads - 1) / threads;

    dim3 block2D(16, 16);
    dim3 grid2D(
        (width + 15) / 16,
        (height + 15) / 16
    );

    CUDA_CHECK(cudaEventRecord(start));

    grayscaleKernel<<<blocks, threads>>>(
        d_input, d_gray, totalPixels
    );

    gaussianKernel<<<grid2D, block2D>>>(
        d_gray, d_blur, width, height
    );

    sobelKernel<<<grid2D, block2D>>>(
        d_blur, d_output, width, height
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed, start, stop));

    totalGpuMs += elapsed;

    std::vector<unsigned char> result(grayBytes);

    CUDA_CHECK(cudaMemcpy(
        result.data(), d_output, grayBytes,
        cudaMemcpyDeviceToHost
    ));

    // Store grayscale edge output as an RGB PPM image.
    std::vector<unsigned char> rgbOutput(rgbBytes);

    for (int i = 0; i < totalPixels; i++) {
        rgbOutput[i * 3] = result[i];
        rgbOutput[i * 3 + 1] = result[i];
        rgbOutput[i * 3 + 2] = result[i];
    }

    savePPM(outputPath, rgbOutput, width, height);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_gray));
    CUDA_CHECK(cudaFree(d_blur));
    CUDA_CHECK(cudaFree(d_output));
}

int main(int argc, char* argv[]) {

    std::string inputDir = "data/input";
    std::string outputDir = "data/output";

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "--input" && i + 1 < argc) {
            inputDir = argv[++i];
        }
        else if (arg == "--output" && i + 1 < argc) {
            outputDir = argv[++i];
        }
        else {
            std::cerr << "Usage: " << argv[0]
                      << " [--input DIR] [--output DIR]\n";
            return EXIT_FAILURE;
        }
    }

    try {
        int device = 0;
        CUDA_CHECK(cudaGetDevice(&device));

        cudaDeviceProp properties;
        CUDA_CHECK(cudaGetDeviceProperties(&properties, device));

        std::cout << "====================================\n";
        std::cout << "CUDA LARGE-SCALE IMAGE PROCESSING\n";
        std::cout << "====================================\n";
        std::cout << "GPU: " << properties.name << "\n";

        fs::create_directories(outputDir);

        std::vector<fs::path> files;

        for (const auto& entry : fs::directory_iterator(inputDir)) {
            if (entry.is_regular_file() &&
                entry.path().extension() == ".ppm") {
                files.push_back(entry.path());
            }
        }

        std::sort(files.begin(), files.end());

        if (files.empty()) {
            throw std::runtime_error("No PPM images found");
        }

        std::cout << "Images found: " << files.size() << "\n";

        double totalGpuMs = 0.0;
        int processed = 0;

        auto hostStart = std::chrono::high_resolution_clock::now();

        for (const auto& file : files) {
            fs::path outputPath =
                fs::path(outputDir) / file.filename();

            processImage(file, outputPath, totalGpuMs);
            processed++;

            if (processed % 20 == 0 || processed == files.size()) {
                std::cout << "Processed: " << processed
                          << "/" << files.size() << "\n";
            }
        }

        auto hostEnd = std::chrono::high_resolution_clock::now();

        double hostMs =
            std::chrono::duration<double, std::milli>(
                hostEnd - hostStart
            ).count();

        std::cout << "\n========== RESULTS ==========\n";
        std::cout << "Images processed: " << processed << "\n";
        std::cout << "GPU kernel time: " << totalGpuMs << " ms\n";
        std::cout << "Total execution time: " << hostMs << " ms\n";
        std::cout << "Average GPU kernel time per image: "
                  << totalGpuMs / processed << " ms\n";
        std::cout << "Output directory: " << outputDir << "\n";
        std::cout << "Status: SUCCESS\n";

    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << "\n";
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
