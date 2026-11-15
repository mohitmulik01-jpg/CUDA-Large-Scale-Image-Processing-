
# CUDA at Scale – Independent Project

## Large-Scale Image Processing Using CUDA

### 1. Introduction

This project implements a GPU-accelerated image processing
application using CUDA C++. The application processes a
dataset of 200 RGB images using parallel GPU computation.

The objective is to demonstrate how CUDA can perform
image-processing operations efficiently using thousands
of GPU threads.

### 2. Project Objectives

- Process hundreds of images in one execution.
- Implement image-processing algorithms using CUDA kernels.
- Perform grayscale conversion.
- Apply Gaussian blur.
- Detect image edges using the Sobel operator.
- Measure GPU kernel execution time.
- Record processing results and performance measurements.

### 3. Dataset Description

The dataset consists of 200 synthetically generated RGB
images.

Image dimensions: 256 x 256 pixels.
Image format: PPM (P6).
Color channels: RGB.

The images contain gradients, circular patterns and
deterministic noise.

### 4. CUDA Implementation

Three CUDA kernels were implemented.

#### 4.1 Grayscale Conversion

The grayscale kernel converts RGB pixels into grayscale
using weighted RGB intensity values.

Gray = 0.299R + 0.587G + 0.114B

Each CUDA thread processes an individual pixel.

#### 4.2 Gaussian Blur

The Gaussian kernel applies a 3x3 weighted filter to
reduce image noise.

The filter uses the following weights:

1 2 1
2 4 2
1 2 1

The resulting weighted sum is normalized by dividing
by 16.

#### 4.3 Sobel Edge Detection

The Sobel kernel detects edges using horizontal and
vertical gradient operators.

Gradient magnitude:

G = sqrt(Gx² + Gy²)

The resulting intensity represents detected edges.

### 5. GPU Execution Architecture

The application allocates device memory and transfers
input image data from host memory to GPU memory.

CUDA kernels execute sequentially on the GPU:

1. Grayscale conversion.
2. Gaussian blur.
3. Sobel edge detection.

The final processed image is copied back to host memory
and saved in PPM format.

### 6. Experimental Environment

Hardware: NVIDIA Tesla T4.
CUDA Toolkit: 13.0.
Compiler: NVIDIA NVCC.
Programming language: CUDA C++.
Execution platform: Google Colab.

### 7. Experimental Results

The application successfully processed 200 images in
a single execution.

Latest recorded measurements:

GPU kernel execution time: 5.00541 ms.
Total application execution time: 301.859 ms.
Average GPU kernel time per image: 0.025027 ms.
Overall throughput: 662.56 images per second.

The GPU kernel time measures the three CUDA kernels.
Total application time also includes image loading,
memory allocation, memory transfers and file output.

### 8. Challenges and Learning

The project involved understanding GPU memory allocation,
CUDA kernel launches, thread organization and synchronization.

Another challenge was handling image boundaries during
Gaussian filtering and Sobel edge detection.

CUDA events were used to measure kernel execution time.

The project provided practical experience in developing,
compiling and executing GPU-accelerated applications.

### 9. Conclusion

The project demonstrates a working CUDA image-processing
pipeline capable of processing 200 images in one execution.

The implementation combines grayscale conversion,
Gaussian filtering and Sobel edge detection.

The experiment provided practical understanding of
parallel GPU computing and performance measurement.
